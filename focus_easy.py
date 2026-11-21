"""Target expensive plane sweeps with exception-aware overlapping route DP."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from itertools import groupby, permutations
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from global_dp_easy import Grid, evaluator
from kttsp_esa_solver import ROOT, validate
from pareto_easy import BudgetExpired, arrival, write_json

SEED = ROOT / "results/esa/easy_global_20261121T120259Z_619ef74b_0003.json"


def mandatory(grid, a, b):
    threshold = grid.udp.dv_threshold
    speed = max(grid.speed_limits[a], grid.speed_limits[b])
    return (abs(grid.energies[a] - grid.energies[b]) > speed * threshold + threshold**2 / 2
            or grid.plane_cost[a,b] > threshold + 1e-5)


def plane_gap(udp, a, b):
    inclinations = udp.tomato_orbital_parameters[:, 2]
    return abs(math.degrees(inclinations[a] - inclinations[b]))


def retime(grid, path, anchors, prefix=(), extra_edge=None, wait=4., flight=9.,
           max_exceptions=None, fine_grid=None, flight_caps=None):
    """Reserve mandatory exceptions on the entire suffix before expanding labels."""
    pairs = tuple(zip(path, path[1:]))
    required = [mandatory(grid, *pair) for pair in pairs]
    suffix_required = [0] * len(path)
    for i in range(len(pairs)-1, -1, -1):
        suffix_required[i] = suffix_required[i+1] + required[i]
    limit = grid.udp.dv_threshold
    budget = grid.udp.n_dv_exception if max_exceptions is None else max_exceptions
    used = sum(leg[4] > limit for leg in prefix)
    states = {used: (arrival(prefix[-1]) if prefix else 0., tuple(prefix))}
    for i in range(len(prefix), len(pairs)):
        grid.check()
        a, b = pairs[i]
        next_states = {}
        for used, (day, schedule) in states.items():
            if used + suffix_required[i] > budget:
                continue
            allow = (used + 1 + suffix_required[i+1] <= budget
                     and (extra_edge is None or required[i] or (a, b) == extra_edge))
            legs = grid.edges(a, b, day, wait, flight, allow)
            if fine_grid is not None and (a,b) in flight_caps:
                cap = flight_caps[a,b]
                useful_fine = min(wait, 2.75, cap+1.)
                legs += fine_grid.edges(a, b, day, useful_fine, cap, allow)
            for old in anchors.get((a, b), ()):
                if old[2] + 1e-6 >= day:
                    legs += (old,)
            for leg in legs:
                if leg is None:
                    continue
                extra = int(leg[4] > limit)
                finish = arrival(leg)
                if (extra and not allow) or finish > grid.horizon:
                    continue
                count = used + extra
                if count not in next_states or finish < next_states[count][0]:
                    next_states[count] = (finish, schedule + (leg,))
        if not next_states:
            return [], i
        states = {}
        earliest = math.inf
        for count, label in sorted(next_states.items()):
            if label[0] < earliest:
                states[count], earliest = label, label[0]
    return [schedule for _, schedule in sorted(states.values())], len(pairs)


def refine_joint(udp, schedule, deadline, submit, weight, rng, perturb, flight_weight=0.,
                 coherent_shift=None):
    """Couple all epochs while differentiating each two-variable Lambert cost."""
    n = len(schedule)
    start = np.asarray([leg[2:4] for leg in schedule]).flatten()
    pairs = [(leg[0], leg[1]) for leg in schedule]
    limits = np.asarray([udp.dv_threshold_exception + 1e-6 if leg[4] > udp.dv_threshold
                         else udp.dv_threshold for leg in schedule]) - 3e-7
    chronology = np.zeros((n - 1, 2 * n))
    for i in range(n - 1):
        chronology[i, 2*i:2*i+3] = (-1., -1., 1.)
    start_margin = np.tile([.55, .22], n)
    if coherent_shift is not None:
        first, last, offset = coherent_shift
        for i in range(first, last):
            start_margin[2*i:2*i+2] = (max(.55, abs(offset)+.48), .50)
    lower = np.maximum(np.tile([0., udp.min_tof], n), start - start_margin)
    upper = np.minimum(udp.max_time, start + start_margin)
    weights = np.full(2*n, weight / n)
    weights[-2:] += 1.0
    for i, (a,b) in enumerate(pairs):
        if 5.0 <= plane_gap(udp, a, b) <= 7.0 and schedule[i][4] <= udp.dv_threshold:
            weights[2*i+1] += flight_weight
    last_values = None
    last_x = None

    def constraint(x):
        nonlocal last_x, last_values
        if time.monotonic() >= deadline:
            raise BudgetExpired
        if last_x is None or not np.array_equal(last_x, x):
            last_x = x.copy()
            last_values = np.asarray([
                udp.compute_transfer(a, b, float(dep), float(tof))
                for (a, b), (dep, tof) in zip(pairs, x.reshape(n, 2))
            ])
        return (limits - last_values) / 100.

    def jacobian(x):
        nominal = constraint(x)
        partials = np.zeros((n, 2*n))
        for i, (a, b) in enumerate(pairs):
            d, t = map(float, x[2*i:2*i+2])
            h = 1e-6
            partials[i, 2*i] = (limits[i]/100 - nominal[i] -
                                  udp.compute_transfer(a, b, d+h, t)/100) / h
            partials[i, 2*i+1] = (limits[i]/100 - nominal[i] -
                                    udp.compute_transfer(a, b, d, t+h)/100) / h
        return partials

    def accept(x):
        violation = constraint(x)
        if min(violation) < -8e-9 or min(chronology @ x) < -1e-6:
            return
        if x[-2] + x[-1] > arrival(schedule[-1]) + 1e-7:
            return
        candidate = tuple((a, b, float(d), float(t), float(dv))
                          for (a, b), (d, t), dv in zip(pairs, x.reshape(n, 2), last_values))
        submit(candidate)

    if perturb:
        for i in range(n):
            if 5 <= plane_gap(udp, *pairs[i]) <= 7 and rng.random() < .28:
                start[2*i:2*i+2] += [rng.uniform(-.09, .09), rng.uniform(-.045, .045)]
    if coherent_shift is not None:
        first, last, offset = coherent_shift
        # Move several consecutive epochs together. Chronology and the
        # nonlinear Lambert constraints are restored by the coupled solve.
        start[2*first:2*last:2] += offset
    start = np.clip(start, lower, upper)
    fit = minimize(lambda x: float(weights @ x), start, jac=lambda _: weights,
                   method="SLSQP", bounds=Bounds(lower, upper),
                   constraints=[LinearConstraint(chronology, -8e-7, np.inf),
                                {"type": "ineq", "fun": constraint, "jac": jacobian}],
                   callback=accept, options={"ftol": 5e-11, "maxiter": 55})
    accept(fit.x)


def neighborhood(grid, path, legs, anchors, left, width, beam, displacement,
                 *, prefix_override=None, coupled=False, score_exit=None):
    """Combine subset labels with a budget reservation and a scored suffix bridge."""
    nodes = path[left:left+width]
    boundary = left+width
    source = path[left-1] if left else None
    target = path[boundary] if boundary < len(path) else None
    prefix = legs[:max(0, left-1)] if prefix_override is None else prefix_override
    after = sum(mandatory(grid, a, b) for a, b in zip(path[boundary:], path[boundary+1:]))
    normal = grid.udp.dv_threshold
    budget = grid.udp.n_dv_exception - after
    start = arrival(prefix[-1]) if prefix else 0.
    used = sum(leg[4] > normal for leg in prefix)
    role_nodes = [*nodes, target] if target is not None else list(nodes)
    indices = {node: i for i, node in enumerate(role_nodes)}
    neighbors = [sum(1 << j for j, b in enumerate(role_nodes) if a != b and not mandatory(grid, a,b))
                 for a in role_nodes]
    all_roles = (1 << len(role_nodes))-1

    @lru_cache(maxsize=None)
    def remaining_exceptions(mask, current):
        # Each component disconnected under the normal threshold must be
        # joined to the others by at least one exceptional edge.
        remaining = (all_roles & ~mask) | (1 << indices[current])
        components = 0
        while remaining:
            queue = remaining & -remaining
            remaining ^= queue
            components += 1
            while queue:
                bit = queue & -queue
                queue ^= bit
                reach = neighbors[bit.bit_length()-1] & remaining
                remaining ^= reach
                queue |= reach
        return components-1

    states = {(0, source, used): (start, (), ())}
    for depth in range(width+int(target is not None)):
        grid.check()
        merged = {}
        for key, (day, segment, local) in states.items():
            mask, last, used = key[:3]
            if depth == width:
                choices = [(None, target)]
            else:
                unvisited = [i for i in range(width) if not mask & (1 << i)]
                choices = [(i, nodes[i]) for i in unvisited[:displacement]]
            for index, b in choices:
                new_mask = mask if index is None else mask | (1 << index)
                needed = remaining_exceptions(new_mask, b)
                if used + needed > budget:
                    continue
                options = grid.edges(last, b, day, 3., 9., used + needed < budget) if last is not None else (None,)
                for old in anchors.get((last, b), ()):
                    if old[2] + 1e-6 >= day:
                        options += (old,)
                for leg in options:
                    if leg is None and last is not None:
                        continue
                    count = used + int(leg is not None and leg[4] > normal)
                    finish = arrival(leg) if leg else day
                    if count + needed > budget or finish > grid.horizon:
                        continue
                    key = (new_mask, b, count)
                    if coupled and (index is None or (depth == width-1 and target is None)):
                        key += (segment,)
                    if key in merged and merged[key][0] <= finish:
                        continue
                    merged[key] = (finish, segment if index is None else segment + (b,),
                                   local + (leg,) if coupled and leg is not None else local)
        if not merged:
            return [], depth
        is_exit = depth == width or (depth == width-1 and target is None)
        if coupled and is_exit:
            def rank(key):
                return (score_exit(merged[key][0], key[2]) if score_exit else merged[key][0],
                        merged[key][0])
            schedules = [(path[:left] + merged[k][1] + path[boundary:], tuple(prefix) + merged[k][2])
                         for k in sorted(merged, key=rank)[:14]]
            return schedules, width
        keys = sorted(merged, key=lambda k: (merged[k][0] + 2.5*k[2], k[0]))
        # Preserve alternatives from several visited subsets instead of
        # letting variants of the same partial permutation fill the beam.
        states = {}
        mask_counts = Counter()
        for key in keys:
            mask_counts[key[0]] += 1
            if mask_counts[key[0]] > max(3, beam//3):
                continue
            states[key] = merged[key]
            if len(states) == beam:
                break
    orders = dict.fromkeys(path[:left] + segment + path[boundary:]
                           for _, segment, _ in states.values() if segment != nodes)
    return list(orders)[:10], width


def shortest_band(grid, a, b, low, high, max_flight):
    """Collect a normal short-flight anchor in a fixed departure band."""
    first = max(0, math.ceil(low/grid.step))
    last = math.floor(high/grid.step)
    if last < first:
        return None
    departures = np.arange(first, last+1)
    a_r, a_v = grid.states(a)[departures].transpose(1, 0, 2)
    state_b = grid.states(b)
    maximum = math.floor(max_flight/grid.step)
    near = []
    for start in range(1, maximum+1, 8):
        grid.check()
        durations = np.arange(start, min(maximum+1, start+8))
        arrivals = departures[None, :] + durations[:, None]
        b_r = state_b[arrivals, 0]
        b_v = state_b[arrivals, 1]
        normal = np.cross(a_r, b_r)
        magnitude = np.linalg.norm(normal, axis=2)
        bound = (np.abs(np.einsum("tdk,dk->td", normal, a_v))
                 + np.abs(np.einsum("tdk,tdk->td", normal, b_v)))
        easy = grid.udp.dv_threshold
        possible = (bound <= (easy+9.)*magnitude) | (magnitude < 1e7)
        found = None
        for i, j in np.argwhere(possible):
            d = int(departures[j])
            f = int(durations[i])
            r1, v1 = grid.ephemeris(a, d)
            r2, v2 = grid.ephemeris(b, d+f)
            flight = f*grid.step
            dep = d*grid.step
            cost = grid.lambert(r1, v1, r2, v2, flight)
            if cost <= easy+9.:
                near.append((flight+.012*max(0., cost-easy), dep, flight))
            if cost <= easy:
                found = (a, b, dep, flight, cost)
                break
        if found:
            break
    else:
        found = None
    choices = sorted(near)[:3]
    for _, dep, flight in choices:
        grid.check()
        fit = minimize(lambda x: x[1], (dep, flight), method="SLSQP",
                       bounds=[(max(low, dep-.16), min(high, dep+.16)),
                               (max(grid.udp.min_tof, flight-.15), min(max_flight, flight+.15))],
                       constraints=[{"type": "ineq", "fun": lambda x: (
                           easy - 1e-6 - grid.udp.compute_transfer(a, b, float(x[0]), float(x[1]))
                       )}], options={"maxiter": 18, "ftol": 1e-10, "eps": 1e-7})
        dep, flight = map(float, fit.x)
        dv = float(grid.udp.compute_transfer(a, b, dep, flight))
        if dv <= easy and (found is None or flight < found[3]):
            found = (a, b, dep, flight, dv)
    return found


def topology_orders(grid, path, legs, left, width, max_required, rng):
    """Explore route order before timing, using only necessary physical bounds."""
    nodes = path[left:left+width]
    prefix = path[:left]
    suffix = path[left+width:]
    outside = sum(mandatory(grid, a, b) for seq in (prefix, suffix) for a, b in zip(seq, seq[1:]))
    budget = max_required - outside
    if budget < 0:
        return []
    known = {leg[:2]: leg[3] for leg in legs}
    allowed = {}
    for a in (*nodes, *(prefix[-1:] if prefix else (None,))):
        for b in (*nodes, *suffix[:1]):
            if a == b:
                continue
            if a is None:
                allowed[a, b] = (0, 0.)
                continue
            if grid.plane_cost[a,b] > grid.udp.dv_threshold_exception + 1e-5:
                continue
            extra = int(mandatory(grid, a, b))
            proxy = known.get((a, b), .7 if extra else .5 + .28*plane_gap(grid.udp, a, b))
            allowed[a, b] = (extra, proxy + rng.random()*.4)
    states = {(0, prefix[-1] if prefix else None, 0): (0., ())}
    for depth in range(width+bool(suffix)):
        grid.check()
        merged = {}
        for (mask, a, used), (score, order) in states.items():
            candidates = [(None, suffix[0])] if depth == width else [
                (i, node) for i, node in enumerate(nodes) if not mask & (1 << i)
            ]
            for index, b in candidates:
                if (a, b) not in allowed:
                    continue
                extra, duration = allowed[a, b]
                count = used + extra
                if count > budget:
                    continue
                new_mask = mask if index is None else mask | (1 << index)
                key = (new_mask, b, count)
                # Keep several distinct permutations per resource state.
                label = (score+duration, order if index is None else order+(b,))
                merged.setdefault(key, []).append(label)
        if not merged:
            return []
        selected = []
        for key, labels in merged.items():
            selected.extend((score, key, order) for score, order in sorted(labels)[:4])
        selected.sort(key=lambda x: (x[0] + .6*x[1][2], x[2]))
        # Include a small number from each mask; this avoids spending all
        # capacity on equivalent paths through an already visited cluster.
        mask_counts = Counter()
        states = {}
        for score, key, order in selected:
            if key in states or mask_counts[key[0]] >= 25:
                continue
            states[key] = (score, order)
            mask_counts[key[0]] += 1
            if len(states) == 240:
                break
        if depth == width or (depth == width-1 and not suffix):
            return list(dict.fromkeys(prefix+order+suffix for _, _, order in selected))
    return []


def recombine_paths(paths):
    """Join orders only when both sides agree on the set already visited."""
    proposals = dict.fromkeys(paths)
    for left in paths:
        prefix_set = set()
        for i, node in enumerate(left):
            prefix_set.add(node)
            for right in paths:
                if right[i] == node and prefix_set == set(right[:i+1]):
                    proposals[left[:i+1]+right[i+1:]] = None
    return list(proposals)


def run(args):
    module = evaluator()
    udp = module.udp
    original = json.loads(args.seed_file.read_text())
    baseline = validate(udp, original["path"], original["transfers"])
    if not baseline or not baseline["valid"]:
        raise ValueError("Seed failed official validation")
    started = time.monotonic()
    deadline = started + args.seconds
    rng = random.Random(args.random_seed)
    best = tuple((*leg[:4], float(udp.compute_transfer(*leg[:4]))) for leg in original["transfers"])
    best_days = baseline["fitness"][0]
    seen = {tuple(leg[:4] for leg in best)}
    stats = Counter()
    logs = []
    reserve = {}
    grids = {}
    last_record = started
    best_changed = math.inf
    atlas = {}
    atlas_rows = []
    attempted = set()
    empty_batches = 0
    shared = {}
    for filename in args.anchor_seed:
        saved = json.loads(filename.read_text())
        certified = validate(udp, saved["path"], saved["transfers"])
        if not certified or not certified["valid"]:
            raise ValueError(f"Invalid anchor schedule {filename}")
        schedule = tuple((*leg[:4], float(udp.compute_transfer(*leg[:4]))) for leg in saved["transfers"])
        reserve[tuple(saved["path"])] = schedule
        for leg in schedule:
            shared.setdefault(leg[:2], []).append(leg)
    for filename in args.flight_atlas:
        data = json.loads(filename.read_text())
        for row in data["flight_atlas"]:
            for leg in row["anchors"]:
                measured = float(udp.compute_transfer(*leg[:4]))
                if measured <= udp.dv_threshold:
                    shared.setdefault(tuple(leg[:2]), []).append((*leg[:4], measured))
    parents = recombine_paths([tuple(original["path"]), *reserve])
    if args.mode == "topology":
        historical = ROOT/"results/esa/easy_pareto_20261121T113857Z_682b68cf_0002.json"
        history = json.loads(historical.read_text())
        historical_legs = tuple((*leg[:4], float(udp.compute_transfer(*leg[:4]))) for leg in history["transfers"])
        if validate(udp, history["path"], historical_legs)["valid"]:
            reserve[tuple(history["path"])] = historical_legs
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = ROOT / "results/esa"
    print(f"{args.mode} | {run_id} | baseline {best_days:.12f}", flush=True)

    def submit(schedule, trial):
        nonlocal best, best_days, last_record, best_changed
        signature = tuple(leg[:4] for leg in schedule)
        if signature in seen:
            return
        seen.add(signature)
        path = [schedule[0][0], *(leg[1] for leg in schedule)]
        stats["checked"] += 1
        checked = validate(udp, path, schedule)
        if not checked or not checked["valid"]:
            stats["rejected"] += 1
            return
        stats["feasible"] += 1
        days = checked["fitness"][0]
        changed = path != original["path"]
        if changed:
            stats["changed_order"] += 1
            best_changed = min(best_changed, days)
        previous = reserve.get(tuple(path))
        if previous is None or days < arrival(previous[-1]):
            reserve[tuple(path)] = schedule
        trial["best_accepted"] = min(trial.get("best_accepted", math.inf), days)
        if days < best_days:
            if best_days - days > 1e-5:
                last_record = time.monotonic()
            best, best_days = schedule, days
            stats["records"] += 1
            file = output/f"easy_focus_{run_id}_{stats['records']:04d}.json"
            write_json(file, {"path": path, "transfers": schedule, "arrival_day": days,
                              "validation": checked, "attempt": trial})
            print(f"ESA record {days:.12f} | {file.name}", flush=True)

    def get_grid(step, phase, refine=None):
        if refine is None:
            refine = args.mode == "exceptions"
        key = step, phase, refine
        if key not in grids:
            grids[key] = Grid(module, step, phase*step, deadline, refine=refine)
            grids[key].horizon = 135.
        return grids[key]

    stop = "time budget"
    try:
        while time.monotonic() < deadline:
            iteration = stats["attempts"]
            if args.mode != "atlas" and iteration >= 8 and time.monotonic() - last_record >= args.stall:
                stop = "stagnation"
                break
            working = best
            if args.mode in ("overlap", "topology", "frontier") and reserve and iteration % 4 == 3:
                choices = sorted(reserve.values(), key=lambda s: arrival(s[-1]))[:8]
                working = rng.choice(choices)
            path = (working[0][0], *(leg[1] for leg in working))
            anchors = {}
            for candidate in (working, best):
                for leg in candidate:
                    anchors.setdefault(leg[:2], []).append(leg)
            for key, choices in atlas.items():
                anchors.setdefault(key, []).extend(choices)
            for key, choices in shared.items():
                anchors.setdefault(key, []).extend(choices)
            targets = sorted(
                ((i, leg) for i, leg in enumerate(working)
                 if 5.0 <= plane_gap(udp, *leg[:2]) <= 7.0 and leg[4] <= udp.dv_threshold),
                key=lambda item: item[1][3], reverse=True,
            )
            trial = {"mode": args.mode, "iteration": iteration}
            logs.append(trial)
            stats["attempts"] += 1
            if args.mode == "clusters":
                # The normal-transition graph largely fixes the high-shell
                # sweeps. Change the order inside each low-shell encounter
                # to shift the launch windows of the high-shell segments.
                low = udp.tomato_orbital_parameters[:,0] < 1e7
                scout = parents[(iteration//3) % len(parents)]
                clusters = [list(items) for is_low, items in groupby(
                    enumerate(scout), key=lambda pair: low[pair[1]]) if is_low]
                index = iteration % len(clusters)
                group = clusters[index]
                left, right = group[0][0], group[-1][0]+1
                phase = (0., .5, .25, .75)[(iteration//3) % 4]
                grid = get_grid(.03125, phase)
                trial.update(cluster=index, window=[left,right], grid_days=grid.step, phase_days=grid.phase)
                for order in permutations(scout[left:right]):
                    route = scout[:left]+order+scout[right:]
                    marker = (route, grid.step, grid.phase)
                    if marker in attempted:
                        continue
                    attempted.add(marker)
                    stats["distinct_route_timing"] += 1
                    timed, _ = retime(grid, route, anchors, wait=5.)
                    spec = {**trial, "visit_order": order}
                    for candidate in timed:
                        submit(candidate, spec)
                    if not timed or arrival(timed[0][-1]) > best_days+.9:
                        continue
                    precise = get_grid(.0078125, .5)
                    marker = (route, precise.step, precise.phase)
                    if marker in attempted:
                        continue
                    attempted.add(marker)
                    stats["distinct_route_timing"] += 1
                    more_anchors = anchors.copy()
                    for leg in timed[0]:
                        more_anchors[leg[:2]] = [*more_anchors.get(leg[:2], ()),leg]
                    fine, _ = retime(precise, route, more_anchors, wait=5.)
                    for candidate in fine:
                        submit(candidate, {**spec, "stage": "fine timing"})
                    if fine and arrival(fine[0][-1]) <= best_days+.25:
                        revised = {**spec, "stage": "joint timing"}
                        refine_joint(udp, fine[0], deadline, lambda s: submit(s, revised), 0., rng, False)
                if iteration >= max(11, 3*len(parents)-1):
                    stop = "cluster order sweep complete"
                    break
            elif args.mode == "mesh":
                step, phase = ((.00390625, 0.), (.00390625, .5),
                               (.0078125, .25), (.00390625, .25))[iteration % 4]
                fine = get_grid(step, phase, refine=iteration % 3 == 2)
                coarse = get_grid(.015625, (.5, 0., .25)[iteration % 3])
                crossings = [leg for leg in working if mandatory(coarse,*leg[:2])]
                long = [leg for _,leg in targets]
                start = (iteration//2*5) % len(long)
                focus = long[start:start+5]
                if iteration % 2 == 0:
                    focus += crossings
                else:
                    focus += [leg for leg in working if leg[3] < .55]
                caps = {leg[:2]: min(9., leg[3]+.55) for leg in focus}
                trial.update(fine_grid_days=step, fine_phase_days=fine.phase, focused_edges=list(caps),
                             local_arc_refinement=fine.refine)
                timed, depth = retime(coarse, path, anchors, wait=5., fine_grid=fine, flight_caps=caps)
                trial["depth"] = depth
                for candidate in timed:
                    submit(candidate, trial)
                if timed and arrival(timed[0][-1]) <= best_days+.28:
                    spec = {**trial, "stage": "joint timing"}
                    refine_joint(udp, timed[0], deadline, lambda s: submit(s, spec), 0., rng, False)
                if iteration >= 13:
                    stop = "selective fine-mesh sweep complete"
                    break
            elif args.mode == "recombine":
                resolution, phase, wait = ((.03125, 0., 6.), (.015625, .5, 6.),
                                           (.015625, 0., 5.), (.0078125, .5, 4.),
                                           (.03125, .25, 8.))[iteration % 5]
                grid = get_grid(resolution, phase)
                order = parents[(iteration//5) % len(parents)]
                trial.update(grid_days=resolution, phase_days=grid.phase, parent_order_index=parents.index(order))
                timed, depth = retime(grid, order, anchors, wait=wait)
                trial["depth"] = depth
                for schedule in timed:
                    submit(schedule, trial)
                if timed and arrival(timed[0][-1]) <= best_days+.30:
                    refined = {**trial, "stage": "joint timing"}
                    refine_joint(udp, timed[0], deadline, lambda s: submit(s, refined), 0., rng, False)
                if iteration >= 5*len(parents)-1:
                    stop = "anchor recombination sweep complete"
                    break
            elif args.mode == "topology":
                grid = get_grid(.0625, (0., .5)[iteration % 2])
                left, right = ((17, 32), (18, 32), (16, 34), (0, 10), (15, 37),
                               (36, 49), (19, 34), (16, 31))[iteration % 8]
                max_required = 4 if iteration % 4 in (0, 1) else 5
                orders = topology_orders(grid, path, working, left, right-left, max_required, rng)
                if iteration < 3:
                    orders += list(reserve)
                trial.update(window=[left,right], max_mandatory_exceptions=max_required, orders=len(orders))
                timed_count = 0
                for order in orders:
                    options = [None]
                    if sum(mandatory(grid, a, b) for a,b in zip(order, order[1:])) <= 4:
                        options.extend([(2,25), (24,6)] if iteration % 2 == 0 else [(22,13), (40,44)])
                    for forced in options:
                        marker = (order, grid.phase, forced)
                        if marker in attempted:
                            continue
                        attempted.add(marker)
                        timed_count += 1
                        stats["distinct_route_timing"] += 1
                        spec = {**trial, "forced_optional_exception": forced}
                        schedules, _ = retime(grid, order, anchors, extra_edge=forced, wait=5.)
                        for schedule in schedules:
                            submit(schedule, spec)
                        if schedules and arrival(schedules[0][-1]) <= best_days+1.:
                            fine = get_grid(.03125, .5)
                            fine_anchors = anchors.copy()
                            for leg in schedules[0]:
                                fine_anchors[leg[:2]] = [*fine_anchors.get(leg[:2], ()), leg]
                            precise, _ = retime(fine, order, fine_anchors, extra_edge=forced, wait=6.)
                            for candidate in precise:
                                submit(candidate, {**spec, "stage": "fine timing"})
                            if precise:
                                refined = {**spec, "stage": "joint timing"}
                                refine_joint(udp, precise[0], deadline, lambda s: submit(s, refined), 0., rng, False)
                        if timed_count >= 15:
                            break
                    if timed_count >= 15:
                        break
                trial["distinct_route_timing"] = timed_count
                empty_batches = 0 if timed_count else empty_batches+1
                if empty_batches >= 36:
                    stop = "no new topology proposals"
                    break
            elif args.mode == "meet":
                split, step, delta = ((8, .03125, .35), (36, .03125, .45),
                                      (20, .015625, .25), (30, .03125, .8),
                                      (42, .03125, .13), (8, .015625, 1.5),
                                      (36, .015625, .10), (20, .03125, 2.5),
                                      (30, .015625, .05), (42, .015625, .9),
                                      (30, .03125, 5.), (20, .03125, 10.))[iteration % 12]
                target = max(101., best_days-delta)
                trial.update(split_index=split, split_tomato=path[split], target_days=target, grid_days=step)
                forward = get_grid(step, (0., .5)[iteration % 2])
                backward = Grid(module, step, forward.phase, deadline, finish_day=target)
                reverse_anchors = {
                    (b,a): [(b,a, target-dep-tof, tof,dv)]
                    for a,b,dep,tof,dv in working if dep+tof <= target
                }
                head, tail = path[:split+1], path[split:][::-1]
                head_required = sum(mandatory(forward, a, b) for a,b in zip(head, head[1:]))
                tail_required = sum(mandatory(backward, a, b) for a,b in zip(tail, tail[1:]))
                early, forward_depth = retime(forward, head, anchors, wait=5.,
                                              max_exceptions=udp.n_dv_exception-tail_required)
                late, backward_depth = retime(backward, tail, reverse_anchors, wait=5.,
                                              max_exceptions=udp.n_dv_exception-head_required)
                trial.update(forward_depth=forward_depth, backward_depth=backward_depth)
                gap = math.inf
                for prefix in early:
                    for suffix in late:
                        if sum(leg[4] > udp.dv_threshold for leg in prefix+suffix) > udp.n_dv_exception:
                            continue
                        wait_margin = target-arrival(suffix[-1])-arrival(prefix[-1])
                        gap = min(gap, -wait_margin)
                        if wait_margin < -.25:
                            continue
                        restored = []
                        for b,a,dep,tof,_ in reversed(suffix):
                            departure = target-dep-tof
                            restored.append((a,b,departure,tof,float(udp.compute_transfer(a,b,departure,tof))))
                        candidate = prefix+tuple(restored)
                        if wait_margin >= -8e-7:
                            submit(candidate, trial)
                        else:
                            stats["near_join_refinements"] += 1
                        refine_joint(udp, candidate, deadline, lambda s: submit(s, trial), 0., rng, False)
                if math.isfinite(gap):
                    trial["minimum_join_gap_days"] = gap
                if iteration >= 11:
                    stop = "split and deadline sweep complete"
                    break
            elif args.mode == "atlas":
                grid = get_grid(.03125, 0.)
                if iteration < len(targets):
                    _, leg = targets[iteration]
                    a, b, dep, flight, _ = leg
                    bands = ((-9., -6.), (-6., -3.), (-3., -1.5),
                             (-1.5, 0.), (0., 1.5), (1.5, 3.))
                    choices = [shortest_band(grid, a, b, max(0., dep+low), dep+high, flight)
                               for low, high in bands if dep+high > 0.]
                    choices = [choice for choice in choices if choice is not None]
                    atlas[a, b] = choices
                    best_flight = min((choice[3] for choice in choices), default=flight)
                    stats["shorter_arcs"] += best_flight < flight-1e-5
                    atlas_rows.append({"edge": [a,b], "incumbent_flight": flight,
                                       "best_independent_flight": min(flight, best_flight),
                                       "anchors": choices})
                else:
                    resolution, phase, wait = ((.03125, 0., 5.), (.015625, 0., 4.),
                                               (.03125, .5, 6.), (.015625, .5, 6.))[iteration % 4]
                    grid = get_grid(resolution, phase)
                    trial.update(grid_days=resolution, phase_days=grid.phase)
                    schedules, depth = retime(grid, path, anchors, wait=wait)
                    trial["depth"] = depth
                    for schedule in schedules:
                        submit(schedule, trial)
                    if schedules and arrival(schedules[0][-1]) <= best_days+.20:
                        refine_joint(udp, schedules[0], deadline, lambda s: submit(s, trial), 0., rng, False)
                    if iteration >= len(targets)+7:
                        stop = "atlas and timing sweep complete"
                        break
            elif args.mode == "joint":
                weight = (0., .025, .10, .30)[iteration % 4]
                flight_weight = (0., .015, .045)[(iteration//4) % 3]
                trial.update(arrival_regularization=weight, plane_sweep_flight_weight=flight_weight)
                refine_joint(udp, working, deadline, lambda s: submit(s, trial), weight, rng,
                             iteration >= 4, flight_weight)
            elif args.mode == "exceptions":
                grid_resolution = (.03125, .03125, .015625, .03125)[iteration % 4]
                grid = get_grid(grid_resolution, (0., .5, .25)[iteration % 3])
                selected = targets[(iteration//2) % len(targets)][1] if iteration % 6 != 0 else None
                forced = selected[:2] if selected else None
                trial.update(grid_days=grid_resolution, phase_days=grid.phase, optional_exception=forced)
                marker = (path, grid_resolution, grid.phase, forced, iteration % 3)
                if marker in attempted:
                    stats["duplicates_skipped"] += 1
                    if stats["duplicates_skipped"] > len(targets)*3:
                        stop = "distinct allocation sweep complete"
                        break
                    continue
                attempted.add(marker)
                schedules, depth = retime(grid, path, anchors, extra_edge=forced,
                                          wait=(4., 6., 3.)[iteration % 3])
                trial["depth"] = depth
                for schedule in schedules:
                    submit(schedule, trial)
                if schedules and arrival(schedules[0][-1]) <= best_days+.15:
                    refined = {**trial, "stage": "joint timing"}
                    refine_joint(udp, schedules[0], deadline, lambda s: submit(s, refined), 0., rng, False)
            else:
                if args.mode == "frontier":
                    left, right = ((0,10), (38,49), (17,31), (18,34), (0,12), (35,49),
                                   (16,34), (18,31), (15,33), (29,44))[iteration % 10]
                    width = right-left
                    chosen = max(working[max(0,left-1):right], key=lambda leg: leg[3])
                    grid_resolution = .0625 if iteration < 7 else .03125
                    beam, displacement = 68, width
                else:
                    target, chosen = targets[(iteration//3) % min(16, len(targets))]
                    width = (12, 15, 18, 21, 14, 17)[iteration % 6]
                    left = min(len(path)-width, max(0, target-width//2+(iteration%3-1)*4))
                    grid_resolution = (.0625, .046875, .0625, .03125, .0625)[iteration % 5]
                    beam, displacement = 20, (4, 6, 5)[iteration % 3]
                grid = get_grid(grid_resolution, (0., .5, .25)[iteration % 3])
                trial.update(window=[left, left+width], focus_edge=chosen[:2],
                             grid_days=grid_resolution, phase_days=grid.phase)
                orders, depth = neighborhood(grid, path, working, anchors, left, width,
                                             beam=beam, displacement=displacement)
                trial.update(depth=depth, orders=len(orders))
                for order in orders:
                    marker = (order, grid_resolution, grid.phase, max(0,left-4))
                    if marker in attempted:
                        continue
                    attempted.add(marker)
                    stats["distinct_route_timing"] += 1
                    prefix = working[:max(0, left-4)]
                    schedules, _ = retime(grid, order, anchors, prefix, wait=4.)
                    for schedule in schedules:
                        submit(schedule, trial)
                    if schedules and arrival(schedules[0][-1]) <= best_days+.25:
                        refined = {**trial, "stage": "joint timing"}
                        refine_joint(udp, schedules[0], deadline, lambda s: submit(s, refined), 0., rng, False)
                if args.mode == "frontier" and iteration >= 19:
                    stop = "frontier sweep complete"
                    break
            if iteration % 4 == 0:
                print(f"{args.mode} | {stats} | best {best_days:.12f} | {time.monotonic()-started:.1f}s", flush=True)
    except BudgetExpired:
        pass
    except KeyboardInterrupt:
        stop = "interrupted"
    finally:
        summary = output/f"easy_focus_{run_id}_summary.json"
        write_json(summary, {"mode": args.mode, "baseline": baseline, "seed": str(args.seed_file),
                             "best_days": best_days, "best_changed_order_days": (
                                 best_changed if math.isfinite(best_changed) else None),
                             "stop": stop, "counts": dict(stats), "attempts": logs,
                             "anchor_seeds": list(map(str,args.anchor_seed)),
                             "anchor_atlases": list(map(str,args.flight_atlas)),
                             "flight_atlas": atlas_rows,
                             "elapsed_seconds": time.monotonic()-started})
        print(f"{args.mode} | stop {stop} | {dict(stats)} | best {best_days:.12f} | {summary.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("joint", "exceptions", "overlap", "atlas", "topology",
                                         "meet", "frontier", "recombine", "mesh", "clusters"))
    parser.add_argument("--seed-file", type=Path, default=SEED)
    parser.add_argument("--anchor-seed", type=Path, action="append", default=[])
    parser.add_argument("--flight-atlas", type=Path, action="append", default=[])
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--stall", type=float, default=240)
    parser.add_argument("--random-seed", type=int, default=501)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 1800:
        parser.error("--seconds must be in (0, 1800]")
    if not math.isfinite(args.stall) or args.stall <= 0:
        parser.error("--stall must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
