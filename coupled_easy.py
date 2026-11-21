"""Bidirectional subset DP and forced-event propagation for ESA Easy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from itertools import permutations
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import differential_evolution, minimize

from focus_easy import mandatory, neighborhood, refine_joint, retime
from global_dp_easy import Grid, evaluator
from kttsp_esa_solver import ROOT, validate
from pareto_easy import BudgetExpired, arrival, write_json

FILES = ROOT / "results/esa"
SEED = FILES / "easy_focus_20261121T131107Z_9a9af22e_0005.json"
ATLAS = FILES / "easy_focus_20261121T131235Z_862089cb_summary.json"
ALTERNATES = (
    FILES / "easy_focus_20261121T130313Z_e97d8883_0008.json",
    FILES / "easy_focus_20261121T130436Z_311bfed4_0009.json",
)


def order_of(legs):
    return (legs[0][0], *(leg[1] for leg in legs))


def exception_count(udp, schedule):
    return sum(leg[4] > udp.dv_threshold for leg in schedule)


def anchors_from(schedules):
    anchors = {}
    for schedule in schedules:
        for leg in schedule:
            bucket = anchors.setdefault(leg[:2], [])
            if leg not in bucket:
                bucket.append(leg)
    return anchors


def restore_suffix(udp, reverse_schedule, finish_day):
    restored = []
    for b, a, dep, tof, _ in reversed(reverse_schedule):
        t = finish_day-dep-tof
        dv = float(udp.compute_transfer(a,b,t,tof))
        restored.append((a,b,t,tof,dv))
    return tuple(restored)


def backward_profiles(module, path, known, targets, deadline, step, phase, budget):
    """Construct complete suffix schedules and index their latest entry times."""
    udp = module.udp
    profiles = []
    if len(path) < 2:
        return [(target, 0, target, ()) for target in targets]
    for target in targets:
        reverse = Grid(module, step, phase, deadline, finish_day=target)
        anchors = {}
        for leg in known:
            a, b, dep, flight, dv = leg
            reverse_dep = target - (dep + flight)
            if reverse_dep >= 0:
                anchors.setdefault((b, a), []).append((b, a, reverse_dep, flight, dv))
        schedules, _ = retime(reverse, path[::-1], anchors, wait=5., max_exceptions=budget)
        for schedule in schedules:
            restored = restore_suffix(udp,schedule,target)
            if any(not math.isfinite(l[4]) or l[4] > udp.dv_threshold_exception+1e-6 for l in restored):
                continue
            profiles.append((restored[0][2], exception_count(udp, restored), arrival(restored[-1]),
                             tuple(restored)))
    return profiles


def middle_orders(grid, path, known, left, right, rng):
    """Enumerate a middle excursion using the necessary Lambert bounds."""
    nodes = path[left:right]
    high = [i for i in nodes if grid.udp.tomato_orbital_parameters[i,0] > 1e7]
    low = tuple(i for i in nodes if i not in high)
    if len(low) != 3:
        return [], 0
    source, target = path[left-1], path[right]
    outside = sum(mandatory(grid,a,b) for sub in (path[:left],path[right:])
                  for a,b in zip(sub,sub[1:]))
    budget = grid.udp.n_dv_exception-outside
    if budget < 2 or budget >= 4:
        return [], 0
    # With a high-shell entry and exit and fewer than four exceptions,
    # all middle low-shell tomatoes must be visited in one excursion.
    mask_all = (1 << (len(high)+1))-1
    block = 1 << len(high)
    possible = {}
    estimates = {}
    limit = grid.udp.dv_threshold_exception
    for a in (source,*nodes):
        for b in (*nodes,target):
            if a == b:
                continue
            energy = abs(grid.energies[a]-grid.energies[b])
            if grid.plane_cost[a,b] > limit+1e-5 or energy > max(grid.speed_limits[a],grid.speed_limits[b])*limit+limit**2/2:
                continue
            extra = int(mandatory(grid,a,b))
            possible[a,b] = extra
            default = .65 if extra else .45+grid.plane_cost[a,b]/28.
            estimates[a,b] = known.get((a,b),default)+rng.random()*.14

    def options(a, mask):
        for j,b in enumerate(high):
            if (mask & 1 << j) and (a,b) in possible:
                yield ((b,), 1 << j, possible[a,b], estimates[a,b])
        if mask & block:
            for group in permutations(low):
                links = tuple(zip((a,*group),group))
                if all(link in possible for link in links):
                    yield group,block,sum(possible[link] for link in links),sum(estimates[link] for link in links)

    @lru_cache(maxsize=None)
    def minimum_left(last, mask):
        """Exact remaining exception count in the contracted static graph."""
        if not mask:
            return possible.get((last,target),budget+1)
        value = budget+1
        for seq,bit,extra,_ in options(last,mask):
            value = min(value, extra+minimum_left(seq[-1],mask^bit))
            if not value:
                break
        if minimum_left.cache_info().misses % 256 == 0:
            grid.check()
        return value

    routes = []
    expansions = 0

    def walk(last, mask, remaining, score, sequence):
        nonlocal expansions
        expansions += 1
        if expansions % 256 == 0:
            grid.check()
        if not mask:
            routes.append((score+estimates[last,target], path[:left]+sequence+path[right:]))
            return
        if expansions >= 90_000:
            return
        for seq,bit,extra,cost in options(last,mask):
            if extra+minimum_left(seq[-1],mask^bit) <= remaining:
                walk(seq[-1],mask^bit,remaining-extra,score+cost,sequence+seq)

    if minimum_left(source,mask_all) <= budget:
        walk(source,mask_all,budget,0.,())
    return [route for _,route in sorted(routes)], expansions


def timed_excursion(grid, fine, path, proposals, left, right, prefix, anchors, finish_cap,
                    diversity=1):
    """Retime the shared subset DAG of all static middle-order proposals."""
    nodes = path[left:right]
    bits = {node: 1 << i for i,node in enumerate(nodes)}
    source, target = path[left-1],path[right]
    full = (1 << len(nodes))-1
    transitions = {}
    for route in proposals:
        mask, last = 0,source
        for node in route[left:right+1]:
            transitions.setdefault((mask,last),set()).add(node)
            mask |= bits.get(node,0)
            last = node
    budget = grid.udp.n_dv_exception
    normal = grid.udp.dv_threshold
    reserved = sum(mandatory(grid,a,b) for a,b in zip(path[right:],path[right+1:]))
    budget -= reserved

    @lru_cache(maxsize=None)
    def remaining(mask,last):
        if last == target:
            return 0 if mask == full else budget+1
        costs = [int(mandatory(grid,last,b))+remaining(mask|bits.get(b,0),b)
                 for b in transitions.get((mask,last),())]
        return min(costs,default=budget+1)

    states = {(0,source,exception_count(grid.udp,prefix),()):(arrival(prefix[-1]),(),())}
    visited = 0
    shells = grid.udp.tomato_orbital_parameters[:,0] > 1e7
    for depth in range(len(nodes)+1):
        grid.check()
        labels = {}
        for (mask,a,used,events),(day,order,chain) in states.items():
            visited += 1
            for b in sorted(transitions.get((mask,a),())):
                added = bits.get(b,0)
                next_mask = mask|added
                need = remaining(next_mask,b)
                if used + need + int(mandatory(grid,a,b)) > budget:
                    continue
                can_spend = used+need < budget
                options = grid.edges(a,b,day,3.5,9.,can_spend)
                if can_spend and mandatory(grid,a,b):
                    options += fine.edges(a,b,day,2.,3.0,True)
                options += tuple(leg for leg in anchors.get((a,b),()) if leg[2]+1e-6 >= day)
                for leg in options:
                    if leg is None:
                        continue
                    count = used+int(leg[4]>normal)
                    finish = arrival(leg)
                    if count+need > budget or finish > finish_cap:
                        continue
                    footprint = events
                    if diversity > 1 and (shells[a] != shells[b] or leg[4] > normal):
                        footprint += ((a,b),)
                    key = (next_mask,b,count,footprint)
                    previous = labels.get(key)
                    if previous is None or finish < previous[0]:
                        labels[key] = (finish,order+((b,) if b!=target else ()),chain+(leg,))
        if not labels:
            return [],visited,depth
        if diversity == 1:
            states = labels
        else:
            groups = Counter()
            states = {}
            for key,label in sorted(labels.items(),key=lambda item: item[1][0]):
                group = key[:3]
                if groups[group] < diversity:
                    states[key] = label
                    groups[group] += 1
    return [(path[:left]+order+path[right:],prefix+chain)
            for _,order,chain in sorted(states.values())],visited,len(nodes)


def optimize_block(udp, pairs, sample, start, deadline, rng):
    """Optimize waits and flights jointly so each shift propagates forward."""
    limits = np.array([udp.dv_threshold_exception+8e-7 if leg[4]>udp.dv_threshold
                       else udp.dv_threshold-3e-7 for leg in sample])
    # Leave a feasible continuous point available whenever the old route is
    # used. The broad ranges let the population jump between phase windows.
    cursor = start
    initial = []
    bounds = []
    for leg in sample:
        wait = max(0.,leg[2]-cursor)
        initial.extend((wait,leg[3]))
        bounds.extend(((0.,max(wait+.55,.9)),
                       (max(udp.min_tof,leg[3]-1.8),leg[3]+.75)))
        cursor = arrival(leg)
    calls = 0
    last_x, last = None, None

    def schedule(x):
        nonlocal calls,last_x,last
        if time.monotonic() >= deadline:
            raise BudgetExpired
        if last_x is not None and np.array_equal(last_x,x):
            return last
        calls += 1
        day = start
        legs = []
        for i,(a,b) in enumerate(pairs):
            dep = day+float(x[2*i])
            flight = float(x[2*i+1])
            dv = float(udp.compute_transfer(a,b,dep,flight))
            legs.append((a,b,dep,flight,dv))
            day = dep+flight
        last_x = x.copy()
        last = tuple(legs)
        return last

    def penalties(x):
        return np.maximum(0.,np.asarray([leg[4] for leg in schedule(x)])-limits)

    def objective(x):
        violation = penalties(x)
        return float(np.sum(x)+.035*np.sum(violation)+.0005*(violation@violation))

    found = differential_evolution(objective,bounds,seed=rng.randrange(2**32),x0=initial,
                                   strategy="best1bin",popsize=6,maxiter=60,polish=False,tol=1e-5)
    fits = [minimize(lambda x: float(np.sum(x)), x, method="SLSQP",
                     bounds=bounds, constraints=[{"type":"ineq","fun":lambda u: (
                         limits-np.asarray([leg[4] for leg in schedule(u)])
                     )}], options={"maxiter":55,"ftol":2e-9,"eps":1e-7})
            for x in (np.asarray(initial),found.x)]
    candidates = []
    for x in (found.x,*(fit.x for fit in fits)):
        legs = schedule(x)
        if all(l[4] <= limit+1e-7 for l,limit in zip(legs,limits)):
            candidates.append(legs)
    return sorted(set(candidates),key=lambda legs:arrival(legs[-1])),calls


class Search:
    def __init__(self, args):
        self.args = args
        self.module = evaluator()
        self.udp = self.module.udp
        self.started = time.monotonic()
        self.deadline = self.started + args.seconds
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
        self.rng = random.Random(args.random_seed)
        self.stats = Counter()
        self.attempts = []
        self.seen = set()
        self.distinct_feasible = set()
        self.tried = set()
        self.grids = {}
        self.baseline, self.best = self.load(args.seed_file)
        self.best_days = self.baseline["fitness"][0]
        self.initial_order = order_of(self.best)
        self.pool = {self.initial_order: self.best}
        self.bank = [self.best]
        extra_files = (*ALTERNATES, *args.anchor_seed)
        for p in extra_files:
            if p.is_file():
                _, alternate = self.load(p)
                self.bank.append(alternate)
                self.pool.setdefault(order_of(alternate), alternate)
        self.atlas = {}
        if args.atlas.is_file():
            for row in json.loads(args.atlas.read_text())["flight_atlas"]:
                for a, b, dep, flight, _ in row["anchors"]:
                    dv = float(self.udp.compute_transfer(a, b, dep, flight))
                    if dv <= self.udp.dv_threshold:
                        self.atlas.setdefault((a,b), []).append((a,b,dep,flight,dv))
        self.seen.add(tuple(x[:4] for x in self.best))

    def load(self, path):
        saved = json.loads(path.read_text())
        result = validate(self.udp, saved["path"], saved["transfers"])
        if not result or not result["valid"]:
            raise ValueError(f"ESA rejected {path}")
        return result, tuple((*leg[:4], float(self.udp.compute_transfer(*leg[:4])))
                             for leg in saved["transfers"])

    def grid(self, step, phase, refine=False):
        key = step, phase, refine
        if key not in self.grids:
            self.grids[key] = Grid(self.module, step, step*phase, self.deadline, refine=refine)
            self.grids[key].horizon = 133.
        return self.grids[key]

    def anchors(self, working):
        anchors = anchors_from((working, self.best, *self.bank))
        for edge, legs in self.atlas.items():
            anchors.setdefault(edge, []).extend(legs)
        return anchors

    def submit(self, candidate, trial, stage):
        key = tuple(leg[:4] for leg in candidate)
        if key in self.seen:
            self.stats["duplicates"] += 1
            return
        self.seen.add(key)
        path = order_of(candidate)
        self.stats["esa_checks"] += 1
        checked = validate(self.udp, path, candidate)
        if not checked or not checked["valid"]:
            self.stats["esa_rejected"] += 1
            return
        self.stats["esa_feasible"] += 1
        fingerprint = tuple((a,b,round(dep,5),round(tof,5)) for a,b,dep,tof,_ in candidate)
        if fingerprint not in self.distinct_feasible:
            self.distinct_feasible.add(fingerprint)
            self.stats["distinct_feasible"] += 1
        days = checked["fitness"][0]
        if path != self.initial_order:
            self.stats["changed_order"] += 1
        trial["best_accepted"] = min(days,trial.get("best_accepted", math.inf))
        old = self.pool.get(path)
        if old is None or days < arrival(old[-1]):
            self.pool[path] = candidate
        if days < self.best_days:
            self.best_days = days
            self.best = tuple(candidate)
            self.stats["records"] += 1
            file = FILES/f"easy_coupled_{self.run_id}_{self.stats['records']:04d}.json"
            write_json(file, {"path": path, "transfers": candidate, "arrival_day": days,
                              "validation": checked, "attempt": {**trial, "stage": stage}})
            print(f"ESA record {days:.12f} | {file.name}", flush=True)

    def finish(self, grid, path, anchors, prefix, trial, stage):
        timed, depth = retime(grid, path, anchors, prefix=prefix, wait=5.5)
        trial["maximum_depth"] = max(trial.get("maximum_depth", 0), depth)
        for schedule in timed:
            self.submit(schedule, trial, stage)
        if timed and arrival(timed[0][-1]) <= self.best_days+.22:
            refine_joint(self.udp, timed[0], self.deadline,
                         lambda s: self.submit(s, trial, stage+" + coupled continuous timing"),
                         0., self.rng, False)

    def window(self, i, working, trial):
        path = order_of(working)
        left, right = ((17,31), (16,32), (18,31), (15,33), (17,34), (19,34),
                       (14,31), (18,35), (13,30), (20,36), (32,49), (0,13))[i%12]
        step, phase = ((.0625, 0.), (.03125, .5), (.046875, .5), (.03125, .25),
                       (.0625, .5), (.03125, 0.))[(i+i//12)%6]
        coarse = self.grid(step,phase)
        fine = self.grid(.015625, (.5,0.,.25)[i%3])
        anchors = self.anchors(working)
        budget = self.udp.n_dv_exception
        mark = (path, left, right, step, phase, tuple(working[max(0,left-4):left]))
        if mark in self.tried:
            return
        self.tried.add(mark)
        trial.update(window=[left,right], grid_days=step, grid_phase_days=coarse.phase)
        self.stats["distinct_windows"] += 1
        required_head = sum(mandatory(coarse, a, b) for a,b in zip(path[:max(0,left-1)],path[1:left]))
        profiles = backward_profiles(self.module, path[right:], (*working, *self.best),
                                     (self.best_days-.12,self.best_days+.55), self.deadline,
                                     .03125, fine.phase, budget-required_head)
        trial["backward_profiles"] = len(profiles)

        def score_exit(day, used):
            choices = [(max(0.,day-release), finish) for release, extra, finish, _ in profiles
                       if used+extra <= budget]
            if choices:
                gap, final = min(choices, key=lambda t: t[0]*3+t[1])
                return final+3*gap
            return self.best_days+20.+.5*used+.1*day

        prefixes = [working[:max(0,left-1)]]
        if left and i%3 != 0:
            required_tail = sum(mandatory(coarse,a,b) for a,b in zip(path[right:],path[right+1:]))
            advanced, _ = retime(fine, path[:left], anchors, wait=5.,
                                 max_exceptions=budget-required_tail)
            for prefix in advanced:
                if tuple(prefix) != tuple(prefixes[0]):
                    prefixes.append(prefix)
        visited = set()
        for prefix in prefixes:
            plans, depth = neighborhood(coarse, path, working, anchors, left, right-left, 64,
                                        right-left, prefix_override=prefix, coupled=True, score_exit=score_exit)
            trial["subset_depth"] = max(depth,trial.get("subset_depth",0))
            for new_order, timed_prefix in plans[:6]:
                prefix_key = (new_order,tuple(l[:4] for l in timed_prefix))
                if prefix_key in visited:
                    continue
                visited.add(prefix_key)
                self.stats["timed_subsets"] += 1
                if right == len(path):
                    self.submit(timed_prefix, trial, "open-end subset schedule")
                else:
                    for dep, used, _, suffix in profiles:
                        if (arrival(timed_prefix[-1]) <= dep+1e-6 and
                                exception_count(self.udp,timed_prefix)+used <= budget):
                            self.submit(timed_prefix+suffix, trial, "bidirectional suffix splice")
                    self.finish(fine, new_order, anchors, timed_prefix, trial, "forward propagation after subset")
                window_anchors = anchors_from((timed_prefix,))
                for edge, legs in anchors.items():
                    window_anchors.setdefault(edge, []).extend(legs)
                self.finish(fine, new_order, window_anchors,
                            timed_prefix[:max(0,left-4)], trial, "coupled full-order retiming")

    def shift(self, i, working, trial):
        path = order_of(working)
        pivot = (29,26,19,8,41,17,27,45,20,37,14,30)[i%12]
        a,b,dep,tof,dv = working[pivot]
        step = (.015625,.0078125,.015625)[i%3]
        fine = self.grid(step,(0.,.5,.25,.75)[(i//12+i)%4])
        reference = self.grid(.015625, (.5,0.,.25)[i%3])
        anchors = self.anchors(working)
        trial.update(pivot=pivot, edge=[a,b], grid_days=step, grid_phase_days=fine.phase)
        mark = (path,pivot,step,fine.phase,round(dep,7))
        if mark in self.tried:
            return
        self.tried.add(mark)
        self.stats["distinct_shift_tests"] += 1
        suffix_min = sum(mandatory(reference,*leg[:2]) for leg in working[pivot+1:])
        normal = self.udp.dv_threshold
        max_used = self.udp.n_dv_exception-suffix_min
        prefixes, pd = retime(reference,path[:pivot+1],anchors, wait=5.5,
                              max_exceptions=max_used-int(mandatory(reference,a,b)))
        trial["prefix_depth"] = pd
        for prefix in prefixes:
            start = arrival(prefix[-1])
            used = exception_count(self.udp,prefix)
            choices = set(anchors.get((a,b), ()))
            for shift in (-1.5,-.95,-.5,-.20,0.,.20,.5,.95):
                earliest = max(start, dep+shift)
                options = fine.edges(a,b,earliest,.11,min(9.,tof+1.20),used < max_used)
                choices.update(l for l in options if l is not None)
            viable = sorted((l for l in choices if l[2]+1e-6 >= start and
                             used+int(l[4]>normal)<=max_used),key=arrival)
            trial["reachable_events"] = len(viable)
            for leg in viable[:10]:
                signature = tuple(x[:4] for x in prefix)+(leg[:4],)
                marker = (path, signature, reference.step, reference.phase)
                if marker in self.tried:
                    self.stats["repeated_events"] += 1
                    continue
                self.tried.add(marker)
                self.stats["propagated_events"] += 1
                detail = {**trial, "forced_departure": leg[2], "forced_flight": leg[3]}
                self.finish(reference,path,anchors,prefix+(leg,),detail,"forced event with suffix propagation")

    def coherent(self, i, working, trial):
        first, last = ((17,30), (24,36), (26,35), (18,27), (6,18),
                       (33,46), (17,48), (25,48), (37,48))[i%9]
        offset = (-.18,-.42,-.93,.24,-.62,.48)[(i//3+i//9)%6]
        marker = (tuple(leg[:4] for leg in working),first,last,offset)
        if marker in self.tried:
            return
        self.tried.add(marker)
        trial.update(transfer_window=[first,last], departure_shift_days=offset)
        self.stats["coherent_trials"] += 1
        refine_joint(self.udp, working, self.deadline,
                     lambda s: self.submit(s,trial,"coherent epoch-shift optimization"),
                     (.02,0.,.10)[i%3], self.rng, False, (.0,.015,0.)[i%3],
                     coherent_shift=(first,last,offset))

    def mirror(self, i, working, trial):
        path = order_of(working)
        left, right = ((17,31), (16,32), (18,31), (15,33),
                       (19,34), (14,31), (18,35), (13,30))[i%8]
        n = len(path)
        target = max(self.best_days,arrival(working[-1])) + (.08,.28,.72)[(i//8+i)%3]
        step = (.0625,.03125,.046875,.03125)[i%4]
        key = (path,left,right,round(target,8),step)
        if key in self.tried:
            return
        self.tried.add(key)
        self.stats["distinct_reverse_windows"] += 1
        reverse = Grid(self.module,step,step/2,self.deadline,finish_day=target)
        reversed_legs = tuple((b,a,target-dep-tof,tof,dv) for a,b,dep,tof,dv in reversed(working))
        reverse_anchors = anchors_from((reversed_legs,))
        forward = self.grid(.015625,(0.,.5,.25)[i%3])
        anchors = self.anchors(working)
        available = self.udp.n_dv_exception
        required_tail = sum(mandatory(forward,*leg[:2]) for leg in working[left-1:])
        heads, _ = retime(forward,path[:left],anchors,wait=5.,
                          max_exceptions=available-required_tail)
        heads.append(working[:left-1])
        trial.update(window=[left,right], reverse_finish_day=target, reverse_grid_days=step,
                     forward_labels=len(heads))

        def score_exit(reverse_arrival, used):
            choices = [max(0.,arrival(head[-1])-target+reverse_arrival) for head in heads
                       if exception_count(self.udp,head)+used <= available]
            return min(choices,default=25.)*6+reverse_arrival

        proposals, depth = neighborhood(reverse,path[::-1],reversed_legs,reverse_anchors,
                                        n-right,right-left,64,right-left,coupled=True,score_exit=score_exit)
        trial["subset_depth"] = depth
        for order, reverse_chain in proposals[:6]:
            actual_order = order[::-1]
            suffix = restore_suffix(self.udp,reverse_chain,target)
            if any(l[4] > self.udp.dv_threshold_exception+1e-6 for l in suffix):
                continue
            self.stats["reverse_timed_subsets"] += 1
            for head in heads:
                if (arrival(head[-1]) <= suffix[0][2]+1e-6 and
                        exception_count(self.udp,head+suffix) <= available):
                    self.submit(head+suffix,trial,"reverse subset with forward-prefix splice")
            merged = anchors_from((suffix,))
            for edge, legs in anchors.items():
                merged.setdefault(edge,[]).extend(legs)
            self.finish(forward,actual_order,merged,working[:left-4],trial,"retiming reverse subset order")

    def excursion(self,i,working,trial):
        path = order_of(working)
        left,right = ((17,31),(16,32),(15,33),(18,31),(16,34),(17,35))[i%6]
        coarse = self.grid(.0625,(0.,.5,.25,.75)[(i//6+i)%4])
        fine = self.grid(.015625,(.5,0.,.25)[i%3])
        precise = self.grid(.0078125,(.5,.25,0.)[i%3])
        known = {}
        for schedule in (working,self.best,*self.bank):
            for leg in schedule:
                known[leg[:2]] = min(known.get(leg[:2],math.inf),leg[3])
        orders, expansions = middle_orders(coarse,path,known,left,right,self.rng)
        trial.update(window=[left,right],static_orders=len(orders),static_expansions=expansions)
        anchors = self.anchors(working)
        prefix = working[:left-4]
        count = 0
        for order in orders:
            marker = (order,coarse.phase,fine.phase,tuple(l[:4] for l in prefix))
            if marker in self.tried:
                continue
            self.tried.add(marker)
            count += 1
            self.stats["excursion_orders_timed"] += 1
            timed, depth = retime(coarse,order,anchors,prefix,wait=4.5)
            trial["maximum_depth"] = max(trial.get("maximum_depth",0),depth)
            for candidate in timed:
                self.submit(candidate,trial,"middle excursion coarse timing")
            if timed and arrival(timed[0][-1]) <= self.best_days+1.8:
                merged = anchors_from((timed[0],))
                for edge,legs in anchors.items():
                    merged.setdefault(edge,[]).extend(legs)
                refined, _ = retime(fine,order,merged,prefix,wait=5.)
                for candidate in refined:
                    self.submit(candidate,trial,"middle excursion fine timing")
                if refined and arrival(refined[0][-1]) <= self.best_days+.9:
                    for leg in refined[0]:
                        merged.setdefault(leg[:2],[]).append(leg)
                    self.finish(precise,order,merged,prefix,trial,"middle excursion precise propagation")
            if count >= 18:
                break
        trial["timed_orders"] = count

    def dag(self,i,working,trial):
        path = order_of(working)
        diversity = {"diverse": 3, "compact": 2}.get(self.args.mode,1)
        if self.args.mode=="compact":
            windows = ((17,31),(16,30),(18,32),(17,32),(19,33),(16,31))
        elif self.args.mode=="diverse":
            windows = ((13,31),(17,34),(14,32),(15,31),(18,33),(18,36))
        else:
            windows = ((17,31),(16,32),(15,33),(18,31),(16,34),(17,35))
        left,right = windows[i%len(windows)]
        if self.args.mode=="compact":
            coarse = self.grid((.0625,.046875)[i%2],(.5,.25,.75,0.)[(i+i//6)%4])
        else:
            coarse = self.grid((.0625,.046875,.03125)[i%3],(0.,.5,.25)[(i+i//6)%3])
        fine = self.grid((.015625,.0078125,.015625)[i%3],(.5,0.,.25)[i%3])
        anchors = self.anchors(working)
        marker = (path,left,right,coarse.step,coarse.phase)
        if marker in self.tried:
            return
        self.tried.add(marker)
        proposals, _ = middle_orders(coarse,path,{leg[:2]:leg[3] for leg in working},left,right,self.rng)
        trial.update(window=[left,right],static_orders=len(proposals),labels_per_state=diversity)
        prefix = working[:left-1]
        cap = working[right][2]+(.5,1.25,2.)[i%3]
        timed,visited,depth = timed_excursion(coarse,fine,path,proposals,left,right,prefix,anchors,cap,
                                              diversity=diversity)
        trial.update(subset_labels=visited, subset_depth=depth, timed_orders=len(timed))
        self.stats["dag_labels"] += visited
        for order,chain in timed:
            self.finish(fine,order,anchors,chain,trial,"shared subset DAG with propagated suffix")
            merged = anchors_from((chain,))
            for edge,legs in anchors.items():
                merged.setdefault(edge,[]).extend(legs)
            self.finish(fine,order,merged,prefix[:max(0,left-4)],trial,"shared subset DAG with coupled retiming")

    def reverse_dag(self,i,working,trial):
        path = order_of(working)
        left,right = ((17,31),(16,32),(15,33),(18,31),(16,34),(17,35))[i%6]
        n = len(path)
        finish = self.best_days+(-.12,.28,-.35,.62)[i%4]
        step = (.0625,.046875,.03125)[i%3]
        reverse = Grid(self.module,step,.5*step,self.deadline,finish_day=finish)
        fine = Grid(self.module,.015625,.0078125,self.deadline,finish_day=finish)
        forward = self.grid(.015625,(.5,0.,.25)[i%3])
        marker = (path,left,right,step,round(finish,8))
        if marker in self.tried:
            return
        self.tried.add(marker)
        proposals,_ = middle_orders(reverse,path,{leg[:2]:leg[3] for leg in working},
                                    left,right,self.rng)
        reversed_anchors = {}
        for a,b,dep,tof,dv in working:
            if dep+tof <= finish:
                reversed_anchors.setdefault((b,a),[]).append((b,a,finish-dep-tof,tof,dv))
        reserved = sum(mandatory(forward,*leg[:2]) for leg in working[:left-1])
        prefixes, depth = retime(fine,path[right:][::-1],reversed_anchors,wait=5.,
                                 max_exceptions=self.udp.n_dv_exception-reserved)
        trial.update(window=[left,right], reverse_finish_day=finish, static_orders=len(proposals),
                     suffix_depth=depth, reverse_release_labels=len(prefixes))
        anchors = self.anchors(working)
        latest = finish-arrival(working[left-2])+1.5
        for prefix in prefixes:
            plans, visited, depth = timed_excursion(reverse,fine,path[::-1],
                                                    [p[::-1] for p in proposals],n-right,n-left,
                                                    prefix,reversed_anchors,latest)
            self.stats["reverse_dag_labels"] += visited
            trial["subset_depth"] = max(trial.get("subset_depth",0),depth)
            for order, reverse_chain in plans:
                suffix = restore_suffix(self.udp,reverse_chain,finish)
                if any(l[4] > self.udp.dv_threshold_exception+1e-6 for l in suffix):
                    continue
                self.stats["reverse_dag_timed_orders"] += 1
                heads, _ = retime(forward,path[:left],anchors,wait=5.,
                                  max_exceptions=self.udp.n_dv_exception-exception_count(self.udp,suffix))
                for head in heads:
                    if arrival(head[-1]) <= suffix[0][2]+1e-6:
                        self.submit(head+suffix,trial,"reverse shared subset DAG with forward splice")
                merged = anchors_from((suffix,))
                for edge,legs in anchors.items():
                    merged.setdefault(edge,[]).extend(legs)
                self.finish(forward,order[::-1],merged,working[:left-4],trial,
                            "reverse shared subset DAG with coupled retiming")

    def timing(self,i,working,trial):
        step,phase = ((.0234375,.5),(.01953125,.25),(.01171875,.5),
                      (.015625,.75),(.0078125,.25),(.013671875,.375),
                      (.009765625,.625),(.015625,.125))[i%8]
        grid = self.grid(step,phase,refine=True)
        trial.update(grid_days=step, grid_phase_days=grid.phase)
        self.stats["adaptive_timing_passes"] += 1
        self.finish(grid,order_of(working),self.anchors(working),(),trial,
                    "full-route DP with continuous window refinement")

    def block(self,i,working,trial):
        left,right = ((25,31),(26,30),(17,23),(24,31),(18,25),(7,12),
                      (29,35),(34,42),(25,30),(40,47))[i%10]
        path = order_of(working)
        anchors = self.anchors(working)
        coarse = self.grid(.03125,(0.,.5,.25)[i%3])
        fine = self.grid(.015625,(.5,.0,.25)[i%3])
        prefix = working[:left]
        if i%3==2:
            budget = self.udp.n_dv_exception-sum(mandatory(coarse,*l[:2]) for l in working[left:])
            earlier,_ = retime(coarse,path[:left+1],anchors,wait=5.,max_exceptions=budget)
            if earlier and arrival(earlier[0][-1]) < arrival(prefix[-1]):
                prefix = earlier[0]
        available = tuple(permutations(path[27:30]))
        if left<=26 and right>=30 and i%2:
            group = available[(i//2+1)%len(available)]
            path = path[:27]+group+path[30:]
        trial.update(transfer_window=[left,right], release_day=arrival(prefix[-1]),
                     internal_order=path[left:right+1])
        segment, calls = optimize_block(self.udp,tuple(zip(path[left:right],path[left+1:right+1])),
                                        working[left:right],arrival(prefix[-1]),self.deadline,self.rng)
        self.stats["block_objective_evaluations"] += calls
        self.stats["feasible_blocks"] += len(segment)
        for block in segment[:3]:
            self.finish(fine,path,anchors,prefix+block,trial,"coupled waits and flights with propagated suffix")

    def run(self):
        print(f"{self.args.mode} | {self.run_id} | ESA baseline {self.best_days:.12f}",flush=True)
        stale = 0
        stop = "time budget"
        try:
            for i in range(self.args.attempts):
                if time.monotonic() >= self.deadline:
                    raise BudgetExpired
                working = self.best
                if self.args.mode in ("window","mirror","excursion","dag","reverse-dag","diverse","compact") and i%5==4:
                    working = self.rng.choice(sorted(self.pool.values(),key=lambda s:arrival(s[-1]))[:5])
                trial = {"mode": self.args.mode, "iteration": i}
                previous = self.stats["distinct_feasible"]
                self.attempts.append(trial)
                if self.args.mode=="window":
                    self.window(i,working,trial)
                elif self.args.mode=="shift":
                    self.shift(i,working,trial)
                elif self.args.mode=="coherent":
                    self.coherent(i,working,trial)
                elif self.args.mode=="mirror":
                    self.mirror(i,working,trial)
                elif self.args.mode=="excursion":
                    self.excursion(i,working,trial)
                elif self.args.mode in ("dag","diverse","compact"):
                    self.dag(i,working,trial)
                elif self.args.mode=="timing":
                    self.timing(i,working,trial)
                elif self.args.mode=="block":
                    self.block(i,working,trial)
                else:
                    self.reverse_dag(i,working,trial)
                stale = stale+1 if self.stats["distinct_feasible"]==previous else 0
                if i%3 == 0:
                    print(f"{self.args.mode} | {dict(self.stats)} | best {self.best_days:.12f} | "
                          f"{time.monotonic()-self.started:.1f}s",flush=True)
                if stale >= self.args.stale:
                    stop = "no new feasible schedules"
                    break
            else:
                stop = "distinct sweep complete"
        except BudgetExpired:
            pass
        except KeyboardInterrupt:
            stop = "interrupted"
        finally:
            file = FILES/f"easy_coupled_{self.run_id}_summary.json"
            write_json(file, {"seed": str(self.args.seed_file), "baseline": self.baseline, "stop": stop,
                              "best_official_days": self.best_days, "counts": dict(self.stats),
                              "attempts": self.attempts, "elapsed_seconds": time.monotonic()-self.started})
            print(f"{self.args.mode} | {stop} | {dict(self.stats)} | best {self.best_days:.12f} | {file.name}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("window","shift","coherent","mirror","excursion","dag",
                                         "timing","block","reverse-dag","diverse","compact"))
    parser.add_argument("--seed-file",type=Path,default=SEED)
    parser.add_argument("--anchor-seed",type=Path,action="append",default=[])
    parser.add_argument("--atlas",type=Path,default=ATLAS)
    parser.add_argument("--seconds",type=float,default=300.)
    parser.add_argument("--attempts",type=int,default=30)
    parser.add_argument("--stale",type=int,default=7)
    parser.add_argument("--random-seed",type=int,default=441)
    args=parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 1800:
        parser.error("--seconds must be in (0, 1800]")
    Search(args).run()


if __name__=="__main__":
    main()
