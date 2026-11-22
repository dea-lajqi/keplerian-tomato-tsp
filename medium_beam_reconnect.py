"""Experimental feasibility-aware segment permutation and downstream retiming for ESA Medium.

Use within kttsp-medium, from the project root. Never writes to baseline/evaluator.
Examples:
 python medium_beam_reconnect.py --seconds 600 --trials 120 --width 6 --beam 20
"""
import argparse
import json
import math
import random
import runpy
import time
from functools import lru_cache
from pathlib import Path

BASE = 'results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json'
EVAL = 'esa_evaluation/spoc-4-keplerian-tomato-traveling-salesperson-medium-1782203984713.py'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default=BASE)
    ap.add_argument('--evaluator', default=EVAL)
    ap.add_argument('--seconds', type=int, default=600)
    ap.add_argument('--trials', type=int, default=120)
    ap.add_argument('--width', type=int, default=6)
    ap.add_argument('--beam', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20261122)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    udp = runpy.run_path(args.evaluator)['udp']
    original = json.loads(Path(args.baseline).read_text())
    order = list(map(int, original['path']))
    rows = original['transfers']
    dep = [float(x[2]) for x in rows]
    tof = [float(x[3]) for x in rows]
    assert len(order) == udp.n_tomatoes and len(rows) == len(order)-1
    f0 = udp.fitness(dep + tof + order)
    assert all(abs(float(x)) < 1e-7 for x in f0[1:]), f0
    baseline_score = float(f0[0]); best_score = baseline_score
    best_order, best_dep, best_tof = order[:], dep[:], tof[:]
    n = len(dep)
    normal = float(udp.dv_threshold)
    exception = float(udp.dv_threshold_exception)
    allowance = int(udp.n_dv_exception)
    deadline = time.monotonic() + args.seconds
    queries = 0

    @lru_cache(maxsize=350000)
    def dv(a, b, t, flight):
        nonlocal queries
        queries += 1
        try:
            return float(udp.compute_transfer(int(a), int(b), float(t), float(flight)))
        except Exception:
            return float('inf')

    def allowed(a, b, t, flight):
        if t < 0 or flight < udp.min_tof or t + flight > udp.max_time: return None
        value = dv(a, b, round(t, 8), round(flight, 8))
        if not math.isfinite(value) or value > exception + 1e-6: return None
        return int(value > normal + 1e-6)

    def candidate_times(earliest, old_dep):
        # Keep existing orbital phases and search around the earliest feasible launch.
        raw = [earliest, old_dep] + [earliest + x for x in (.0625, .125, .25, .5, 1.)] + [old_dep + x for x in (-.75, -.5, -.25, -.125, .125, .25)]
        return sorted({round(t, 8) for t in raw if t >= earliest-1e-8 and t >= 0})

    def candidate_flights(i):
        f = tof[i]
        return sorted({round(max(udp.min_tof, v), 8) for v in
                      [f, f*.5, f*.75, f*.9, f*1.1, f*1.3,
                       f-.75, f-.5, f-.25, f-.125, f+.125, f+.25, f+.5,
                       .0625, .125, .1875, .25, .375, .5, .75, 1., 1.5, 2., 3., 4.]})

    def forward_suffix(start_idx, perm, newdep, newtof, exc_used):
        # Retimes the ENTIRE affected suffix, allowing gains to propagate.
        arrival = newdep[-1] + newtof[-1]
        for i in range(start_idx, n):
            if time.monotonic() >= deadline: return None
            a, b = perm[i], perm[i+1]
            chosen = None
            # Fast path: reuse an existing transfer without extra Lambert work.
            if dep[i] >= arrival-1e-8:
                d = dep[i]; f = tof[i]
                e = allowed(a, b, d, f)
                if e is not None and exc_used + e <= allowance:
                    chosen = (d+f, d, f, e)
            # Try to advance the incumbent departure and flight simultaneously.
            if arrival <= dep[i]+.5:
                for d in candidate_times(arrival, dep[i]):
                    if d > dep[i]+.25: continue
                    for f in candidate_flights(i):
                        finish = d+f
                        if finish >= best_score: continue
                        if chosen is not None and finish >= chosen[0]-1e-8: continue
                        e = allowed(a, b, d, f)
                        if e is None or exc_used+e > allowance: continue
                        chosen = (finish, d, f, e)
            if chosen is None: return None
            finish, d, f, e = chosen
            exc_used += e
            newdep.append(d); newtof.append(f)
            arrival = finish
            if arrival >= best_score: return None
        return newdep, newtof

    def solve_window(left, count):
        # The left and right endpoints stay fixed; intermediate tomatoes may be
        # permuted. Permutation search is combined with feasibility screening.
        right = left + count + 1
        if right >= len(best_order): return None
        interior = best_order[left+1:right]
        prev_arrival = best_dep[left-1] + best_tof[left-1] if left else 0.0
        prefix_exc = sum(float(r[4]) > normal+1e-6 for r in rows[:left])
        # Better use recomputed prefix counts whenever incumbent is updated.
        prefix_exc = 0
        for i in range(left):
            e = allowed(best_order[i], best_order[i+1], best_dep[i], best_tof[i])
            if e is None: return None
            prefix_exc += e
        if prefix_exc > allowance: return None
        # state: (arrival, used_exceptions, last_node, mask, dep_block, tof_block, order_block)
        states = [(prev_arrival, prefix_exc, best_order[left], 0, [], [], [])]
        for depth in range(count+1):
            candidates = []
            for arrive, used, last, mask, ds, fs, os in states:
                available = [(j, v) for j, v in enumerate(interior) if not(mask>>j)&1] if depth < count else [(-1, best_order[right])]
                for j, nxt in available:
                    i = left+depth
                    for t in candidate_times(arrive, best_dep[i]):
                        # If the block is already several days behind, reconnecting
                        # later is unlikely to help; prune the expensive searches.
                        if t > best_dep[i]+1.5: continue
                        for f in candidate_flights(i):
                            fin = t+f
                            if fin > best_dep[min(right,n-1)] + 2.0: continue
                            e = allowed(last, nxt, t, f)
                            if e is None or used+e>allowance: continue
                            candidates.append((fin,used+e,nxt,mask|(1<<j) if j>=0 else mask,
                                               ds+[t],fs+[f],os+[nxt] if j>=0 else os))
                if time.monotonic() >= deadline: return None
            if not candidates: return None
            # Keep diverse arrival phases and exception budgets, not only earliest.
            candidates.sort(key=lambda x: (x[0]+.35*x[1], x[0]))
            seen = set(); states = []
            for item in candidates:
                key = (item[2],item[3],item[1],round(item[0]*8))
                if key in seen: continue
                seen.add(key); states.append(item)
                if len(states)>=args.beam: break
        return right, states

    out = Path('results/esa/medium_beam_reconnect')
    out.mkdir(parents=True, exist_ok=True)
    checked = 0; complete = 0
    print(f'ESA baseline {baseline_score:.9f} days; {n+1} tomatoes; {allowance} exceptions', flush=True)
    for trial in range(1,args.trials+1):
        if time.monotonic() >= deadline: break
        count = rng.randint(3, max(3,args.width))
        left = rng.randrange(0, n-count)
        trial_solution = solve_window(left,count)
        if trial_solution is None: continue
        right, states = trial_solution
        for fin,used,last,mask,ds,fs,block_order in states:
            cand_order = best_order[:left+1] + block_order + best_order[right:]
            if len(cand_order) != n+1 or len(set(cand_order))!=n+1: continue
            dnew = best_dep[:left] + ds
            fnew = best_tof[:left] + fs
            checked += 1
            completed = forward_suffix(right, cand_order, dnew, fnew, used)
            if completed is None: continue
            complete += 1
            dnew, fnew = completed
            if dnew[-1]+fnew[-1] >= best_score-1e-7: continue
            fit = udp.fitness(dnew+fnew+cand_order)
            if abs(float(fit[1])) > 1e-7 or any(float(x)>1e-7 for x in fit[2:]): continue
            score = float(fit[0])
            if score >= best_score-1e-7: continue
            best_score=score
            best_dep,best_tof,best_order=dnew[:],fnew[:],cand_order[:]
            newlegs=[]
            for i in range(n):
                newlegs.append([cand_order[i],cand_order[i+1],dnew[i],fnew[i],
                                dv(cand_order[i],cand_order[i+1],round(dnew[i],8),round(fnew[i],8))])
            result = {'path':cand_order,'transfers':newlegs,'arrival_day':score,
                      'exceptions':sum(x[4]>normal+1e-6 for x in newlegs),
                      'validation':{'fitness':list(map(float,fit)),'valid':True},
                      'method':'beam_reconnect_medium','baseline':args.baseline,'trial':trial}
            file=out/f'validated_{score:.9f}_trial{trial}.json'
            file.write_text(json.dumps(result,indent=2))
            print('NEW ESA-VALID BEST:',round(score,9),'file:',file,flush=True)
        if trial%10==0:
            print(f'Trial {trial}, screened {checked}, complete suffixes {complete}, Lambert calls {queries}, '
                  f'best {best_score:.9f}, elapsed {args.seconds-max(0,deadline-time.monotonic()):.1f}s',flush=True)
    print(f'Finished. Best ESA-validated result: {best_score:.9f} days; '
          f'improvement {baseline_score-best_score:.9f} days; screened {checked}; '
          f'complete suffixes {complete}; Lambert calls {queries}',flush=True)

if __name__=='__main__':main()
