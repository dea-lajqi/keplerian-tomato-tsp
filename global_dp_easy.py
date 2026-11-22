"""Full-route retiming and large permutation search for ESA Easy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import minimize

from kttsp_esa_solver import ROOT, validate
from pareto_easy import BudgetExpired, Transfers, arrival, write_json

SEED = ROOT / "results/esa/easy_pareto_20261121T113857Z_682b68cf_0002.json"
EVALUATOR = ROOT / "esa_evaluation/spoc-4-keplerian-tomato-traveling-salesperson-small-1782203910047.py"


def evaluator():
    spec = importlib.util.spec_from_file_location("esa_original_small", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.pk.version != "2.6.4":
        raise RuntimeError("This search requires PyKEP 2.6.4")
    return module


def plane_cost_bound(elements, mu):
    """Lower bound for rotating orbital angular momentum with two impulses."""
    axes, eccentricities, inclination, raan = elements[:, :4].T
    normals = np.column_stack((np.sin(inclination)*np.sin(raan),
                               -np.sin(inclination)*np.cos(raan), np.cos(inclination)))
    cosine = np.clip(normals @ normals.T, 0., 1.)
    # The local transverse speed is smallest at apoapsis. For a rotation
    # less than 90 degrees the cheapest burn has magnitude u*sin(angle).
    # Beyond 90 degrees the infimum is u. The two plane angles together
    # must span the angle between the endpoint orbit normals.
    speed = np.sqrt(mu*(1.-eccentricities)/(axes*(1.+eccentricities)))
    return np.minimum.outer(speed, speed)*np.sqrt(1.-cosine*cosine)


class Grid:
    """Search a time lattice, pruning with a lower bound on plane-change cost."""

    def __init__(self, module, step, phase, deadline, refine=False, finish_day=None, directional=False):
        self.module = module
        self.udp = module.udp
        self.step = step
        self.phase = phase
        self.deadline = deadline
        self.refine = refine
        self.finish_day = finish_day
        self.directional = directional
        self.horizon = self.udp.max_time if finish_day is None else finish_day
        self.stats = Counter()
        axes, eccentricities = self.udp.tomato_orbital_parameters[:, :2].T
        self.energies = -module.MU_MOON / (2 * axes)
        self.speed_limits = np.sqrt(module.MU_MOON * (1 + eccentricities) / (axes * (1 - eccentricities)))
        self.z_speed = (np.sqrt(module.MU_MOON*axes*(1-eccentricities**2))
                        *np.cos(self.udp.tomato_orbital_parameters[:,2])/(axes*(1+eccentricities)))
        self.plane_cost = plane_cost_bound(self.udp.tomato_orbital_parameters, module.MU_MOON)
        self.ephemeris = lru_cache(maxsize=130_000)(self._ephemeris)
        self.positions = {}
        self.options = lru_cache(maxsize=40_000)(self._options)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired

    def epoch(self, tick):
        return tick * self.step + self.phase

    def _ephemeris(self, node, tick):
        return self.state_at(node, self.epoch(tick))

    def state_at(self, node, day):
        planet = self.udp.tomatoes[node]
        if self.finish_day is None:
            return planet.eph(day)
        position, velocity = planet.eph(self.finish_day - day)
        return position, tuple(-component for component in velocity)

    def states(self, node):
        if node not in self.positions:
            count = math.ceil(self.horizon / self.step) + 2
            self.positions[node] = np.asarray([self.ephemeris(node, tick) for tick in range(count)])
        return self.positions[node]

    def lambert(self, r1, v1, r2, v2, flight, senses=(False, True)):
        self.stats["lambert_samples"] += 1
        best = math.inf
        for cw in senses:
            self.stats["lambert_orientations"] += 1
            try:
                sol = self.module.pk.lambert_problem(
                    r1, r2, flight * self.module.pk.DAY2SEC, self.module.MU_MOON, cw, self.udp.max_revs
                )
            except (RuntimeError, ValueError):
                continue
            for start, end in zip(sol.get_v1(), sol.get_v2()):
                best = min(best, math.dist(start, v1) + math.dist(end, v2))
        return best

    def _options(self, a, b, first, max_wait, max_flight, allow_exception):
        self.check()
        self.stats["arc_queries"] += 1
        last = min(first + math.floor(max_wait / self.step), math.floor((self.horizon - self.phase) / self.step) - 1)
        final = min(last + math.floor(max_flight / self.step), math.floor((self.horizon - self.phase) / self.step))
        if first > last:
            return (None, None)
        states_a, states_b = self.states(a), self.states(b)
        deps = np.arange(first, last + 1)
        a_r, a_v = states_a[deps].transpose(1, 0, 2)
        if self.directional:
            a_h = np.cross(a_r,a_v)/np.linalg.norm(a_r,axis=1)[:,None]
        record = [None, None]
        normal_limit = self.udp.dv_threshold
        special_limit = self.udp.dv_threshold_exception if allow_exception else normal_limit
        energy_gap = abs(self.energies[a] - self.energies[b])
        speed = max(self.speed_limits[a], self.speed_limits[b])
        # Two burns of total magnitude D can change orbital energy by at
        # most max(endpoint speeds)*D + D²/2, measured from the given orbits.
        normal_possible = (energy_gap <= speed * normal_limit + normal_limit**2 / 2
                           and self.plane_cost[a,b] <= normal_limit + 1e-5)
        if (energy_gap > speed * special_limit + special_limit**2 / 2
                or self.plane_cost[a,b] > special_limit + 1e-5):
            self.stats["physically_pruned_arcs"] += 1
            return (None, None)
        starts = [{}, {}]
        # Process contiguous arrival batches in order. The first accepted leg
        # of each class attains that class's earliest lattice arrival.
        batch = min(32, max(8, int(0.5 / self.step)))
        duration_max = math.floor(max_flight / self.step)
        for start in range(first + 1, final + 1, batch):
            self.check()
            ends = np.arange(start, min(start + batch, final + 1))
            valid = (ends[:, None] > deps) & (ends[:, None] - deps <= duration_max)
            normal = np.cross(a_r, states_b[ends, 0, None, :])
            magnitude = np.linalg.norm(normal, axis=2)
            a_perp = np.einsum("edk,dk->ed", normal, a_v)
            b_perp = np.einsum("edk,ek->ed", normal, states_b[ends, 1])
            cost = np.abs(a_perp)+np.abs(b_perp)
            limit = normal_limit if record[1] else special_limit
            if self.refine:
                limit += 14.0
            possible = valid & ((cost <= (limit + 1e-5) * magnitude) | (magnitude < 1e7))
            static_senses=(False,True)
            if self.directional:
                # A burn of size d can alter h_z by at most r_apo*d.
                # A Lambert direction with the wrong h_z sign must at
                # least cancel the corresponding endpoint component.
                za,zb=self.z_speed[a],self.z_speed[b]
                static_senses=tuple(cw for cw,lower in (
                    (False,max(0.,-za)+max(0.,-zb)),
                    (True,max(0.,za)+max(0.,zb)),
                ) if lower<=limit+1e-4)
            dynamic_direction=self.directional and len(static_senses)==2
            if not static_senses:
                continue
            if dynamic_direction:
                b_r,b_v = states_b[ends].transpose(1,0,2)
                b_h = np.cross(b_r,b_v)/np.linalg.norm(b_r,axis=1)[:,None]
                at = np.einsum("edk,dk->ed",normal,a_h)
                bt = np.einsum("edk,ek->ed",normal,b_h)
                allowed_cost = (limit+1e-4)*magnitude
                plus = (np.hypot(a_perp,np.minimum(0.,at))
                        +np.hypot(b_perp,np.minimum(0.,bt)))<=allowed_cost
                minus = (np.hypot(a_perp,np.maximum(0.,at))
                         +np.hypot(b_perp,np.maximum(0.,bt)))<=allowed_cost
                ambiguous = (magnitude<1e7) | (np.abs(normal[:,:,2])<1e-8*magnitude)
                possible &= plus|minus|ambiguous
            self.stats["grid_cells"] += int(np.count_nonzero(valid))
            for ei, di in np.argwhere(possible):
                if cost[ei, di] > (limit + 1e-5) * magnitude[ei, di] and magnitude[ei, di] >= 1e7:
                    continue
                d, e = int(deps[di]), int(ends[ei])
                dep = self.epoch(d)
                # Construct ephemerides at the exact binary departure+flight,
                # including for shifted grids. The bound above has slack.
                flight = (e - d) * self.step
                r1, v1 = self.ephemeris(a, d)
                r2, v2 = (self.ephemeris(b, e) if dep + flight == self.epoch(e)
                          else self.state_at(b, dep + flight))
                senses=static_senses
                if dynamic_direction and not ambiguous[ei,di] and plus[ei,di]!=minus[ei,di]:
                    senses=(bool(plus[ei,di])!=bool(normal[ei,di,2]>0.),)
                dv = self.lambert(r1, v1, r2, v2, flight, senses)
                if self.refine:
                    for category, threshold in enumerate((normal_limit, special_limit)):
                        if (category == 0 and not normal_possible) or (category == 1 and not allow_exception):
                            continue
                        if dv > threshold + 14 or record[category] is not None:
                            continue
                        cell = (d // 4, (e - d) // 4)
                        merit = dep + flight + max(0., dv - threshold) * .006
                        if cell not in starts[category] or merit < starts[category][cell][0]:
                            starts[category][cell] = (merit, dep, flight)
                if dv <= special_limit:
                    extra = int(dv > normal_limit)
                    if record[extra] is None:
                        record[extra] = (a, b, dep, flight, dv)
                if record[0] or (not normal_possible and record[1]):
                    # Once a normal leg arrives, later expensive legs are dominated.
                    if self.refine:
                        return self.refine_options(record, starts, a, b, first, last, max_flight)
                    return tuple(record)
        if self.refine:
            return self.refine_options(record, starts, a, b, first, last, max_flight)
        return tuple(record)

    def refine_options(self, record, starts, a, b, first, last, max_flight):
        for category, threshold in enumerate((self.udp.dv_threshold, self.udp.dv_threshold_exception)):
            choices = sorted(starts[category].values())[:4]
            if record[category]:
                choices.append((0., *record[category][2:4]))
            for _, departure, flight in choices:
                self.check()
                self.stats["continuous_arc_solves"] += 1
                limit = threshold - 1e-6
                bounds = [
                    (max(self.epoch(first), departure - 3*self.step), min(self.epoch(last), departure + 3*self.step)),
                    (max(self.udp.min_tof, flight - 3*self.step), min(max_flight, flight + 3*self.step)),
                ]
                fit = minimize(lambda x: x[0] + x[1], (departure, flight), method="SLSQP",
                               bounds=bounds, constraints=[{"type": "ineq", "fun": lambda x: (
                                   limit - self.udp.compute_transfer(a, b, float(x[0]), float(x[1]))
                               )}], options={"maxiter": 14, "ftol": 1e-8, "eps": 1e-7})
                dep, tof = map(float, fit.x)
                if record[category] and dep + tof >= arrival(record[category]):
                    continue
                dv = float(self.udp.compute_transfer(a, b, dep, tof))
                if dv <= threshold:
                    actual = int(dv > self.udp.dv_threshold)
                    if record[actual] is None or dep + tof < arrival(record[actual]):
                        record[actual] = (a, b, dep, tof, dv)
        return tuple(record)

    def edges(self, a, b, day, wait, flight, allow_exception=True):
        first = max(0, math.ceil((day - 9e-7 - self.phase) / self.step))
        return self.options(a, b, first, wait, flight, allow_exception)


def retime(grid, path, old, wait, flight, prefix=()):
    """Retiming DP with one earliest-arrival label for each exception count."""
    normal = grid.udp.dv_threshold
    budget = grid.udp.n_dv_exception
    used = sum(leg[4] > normal for leg in prefix)
    states = {used: (arrival(prefix[-1]) if prefix else 0.0, tuple(prefix))}
    boundary = len(prefix)
    depth = boundary
    for a, b in zip(path[boundary:], path[boundary + 1:]):
        grid.check()
        next_states = {}
        old_leg = old.get((a, b))
        for used, (day, schedule) in states.items():
            edges = grid.edges(a, b, day, wait, flight, used < budget)
            if old_leg and old_leg[2] + 1e-6 >= day:
                edges += (old_leg,)
            for leg in edges:
                if not leg:
                    continue
                count = used + int(leg[4] > normal)
                finish = arrival(leg)
                if count <= budget and (count not in next_states or finish < next_states[count][0]):
                    next_states[count] = (finish, schedule + (leg,))
        if not next_states:
            return [], depth
        # A label reached earlier with fewer exceptions dominates later labels.
        states = {}
        earliest = math.inf
        for count, label in sorted(next_states.items()):
            if label[0] < earliest:
                states[count] = label
                earliest = label[0]
        depth += 1
    return [schedule for _, schedule in sorted(states.values())], depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--seed-file", type=Path, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/esa")
    parser.add_argument("--random-seed", type=int, default=101)
    parser.add_argument("--strategy", choices=("mixed", "adaptive", "routes", "reverse-time", "wide"), default="mixed")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 900:
        parser.error("--seconds must be in (0, 900]")
    run(args)


def run(args):
    module = evaluator()
    udp = module.udp
    seed = json.loads(args.seed_file.read_text(encoding="utf-8"))
    base = validate(udp, seed["path"], seed["transfers"])
    if not base or not base["valid"]:
        raise ValueError("Invalid ESA seed")
    started = time.monotonic()
    deadline = started + args.seconds
    stats = Counter()
    best_days = base["fitness"][0]
    best = tuple((*leg[:4], float(udp.compute_transfer(*leg[:4]))) for leg in seed["transfers"])
    original = tuple(seed["path"])
    best_route = original
    best_changed = math.inf
    seen = {tuple(leg[:4] for leg in best)}
    routes = set()
    rng = random.Random(args.random_seed)
    report = []
    aggregates = Counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"ESA baseline {base['fitness']} | {run_id}", flush=True)

    def submit(schedule, attempt):
        nonlocal best_days, best, best_route, best_changed
        signature = tuple(leg[:4] for leg in schedule)
        if signature in seen:
            return
        seen.add(signature)
        path = (schedule[0][0], *(leg[1] for leg in schedule))
        stats["esa_checks"] += 1
        checked = validate(udp, path, schedule)
        if not checked or not checked["valid"]:
            stats["esa_rejected"] += 1
            return
        stats["esa_feasible"] += 1
        routes.add(path)
        days = checked["fitness"][0]
        if path != original:
            stats["changed_order"] += 1
            if days < best_changed:
                best_changed = days
                if days < base["fitness"][0] and days >= best_days:
                    file = args.output_dir / f"easy_global_{run_id}_alternate_{stats['changed_order']:04d}.json"
                    write_json(file, {"path": path, "transfers": schedule, "arrival_day": days,
                                      "validation": checked, "attempt": attempt})
        attempt.setdefault("valid_days", []).append(days)
        if days < best_days:
            best_days, best, best_route = days, tuple(schedule), path
            stats["records"] += 1
            file = args.output_dir / f"easy_global_{run_id}_{stats['records']:04d}.json"
            write_json(file, {"path": path, "transfers": schedule, "arrival_day": days,
                              "validation": checked, "attempt": attempt})
            print(f"ESA record {days:.12f} | {file.name}", flush=True)

    grids = {}
    try:
        for iteration in range(1_000_000):
            if time.monotonic() >= deadline:
                raise BudgetExpired
            # First measure full-route retiming and the reversed route. Then
            # interleave whole-order moves with multi-node permutation DP.
            steps = ((.03125, 0), (.03125, 0), (.015625, 0), (.015625, .5), (.0078125, 0))
            if args.strategy == "adaptive":
                steps = ((.0625, 0), (.0625, .5), (.03125, 0), (.03125, .5), (.03125, .25))
            elif args.strategy == "routes":
                steps = ((.0625, 0), (.0625, 0), (.0625, .5), (.0625, .25), (.03125, 0))
            elif args.strategy == "wide":
                steps = ((.0625, 0), (.0625, .5), (.03125, 0), (.0625, .25), (.0625, .75))
            step, phase_fraction = steps[iteration % 5]
            key = (step, phase_fraction)
            if key not in grids:
                grids[key] = Grid(module, step, step * phase_fraction, deadline, args.strategy == "adaptive")
            grid = grids[key]
            old = {(leg[0], leg[1]): leg for leg in best}
            wait, flight = (3.0, 10.0) if iteration % 4 else (5.0, 12.0)
            attempt = {"step_days": step, "phase_days": grid.phase}
            report.append(attempt)
            if args.strategy == "reverse-time":
                offset = (.05, .5, 1.5, 4., .125, 2.5, 8., 19.)[iteration % 8]
                finish_day = max(101., best_days - offset)
                path = best_route if iteration % 3 else structural_route(best_route, udp, rng, iteration)
                reverse_grid = Grid(module, step, step * phase_fraction, deadline, finish_day=finish_day)
                try:
                    anchors = {(b, a): (b, a, max(0., finish_day - dep - tof), tof, dv)
                               for a, b, dep, tof, dv in best if dep + tof <= finish_day}
                    attempt.update(method="deadline-driven backward timing DP", path=path, deadline_day=finish_day)
                    stats["attempts"] += 1
                    schedules, depth = retime(reverse_grid, path[::-1], anchors, wait, flight)
                    attempt["completed_depth"] = depth
                    for schedule in schedules:
                        candidate = []
                        for b, a, dep, tof, dv in reversed(schedule):
                            departure = max(0., finish_day - dep - tof)
                            candidate.append((a, b, departure, tof, float(udp.compute_transfer(a, b, departure, tof))))
                        submit(candidate, attempt)
                finally:
                    aggregates.update(reverse_grid.stats)
                continue
            if args.strategy == "wide":
                width = rng.choice((12, 14, 16, 18, 22))
                focus = rng.choice(sorted(range(len(best)), key=lambda i: best[i][3], reverse=True)[:16])
                left = min(len(best_route) - width, max(0, focus - rng.randrange(width)))
                attempt.update(method="displacement-bounded permutation DP", window=[left, left + width])
                stats["attempts"] += 1
                orders, depth = reorder(grid, best_route, best, old, left, width, wait, flight, displacement=4)
                attempt["completed_depth"] = depth
                for path in orders:
                    prefix = best[:max(0, left - 3)]
                    timed, _ = retime(grid, path, old, wait, flight, prefix)
                    for schedule in timed:
                        submit(schedule, attempt)
                    if timed and arrival(timed[0][-1]) <= best_days + .25:
                        refined = Transfers(udp, deadline).polish(timed[0], 0.)
                        submit(refined, {**attempt, "method": f"{attempt['method']} + continuous refinement"})
                print(f"wide window {left}:{left+width}, depth {depth}, generated {len(orders)} orders | "
                      f"best {best_days:.12f}", flush=True)
                continue
            if args.strategy == "routes" and iteration % 5 in (2, 3, 4):
                path = structural_route(best_route, udp, rng, iteration)
                prefix, method = (), "shell-cluster permutation with full timing DP"
            elif iteration in (0, 2, 3, 4) or iteration % 11 == 0:
                path, prefix, method = best_route, (), "whole-route timing DP"
            elif iteration == 1 or iteration % 17 == 0:
                path, prefix, method = best_route[::-1], (), "route reversal with full timing DP"
            elif iteration % 4 == 1:
                method = "large permutation DP"
                width = rng.choice((10, 11, 12, 13, 14))
                left = rng.randrange(len(best_route) - width + 1)
                attempt.update(method=method, window=[left, left + width])
                stats["attempts"] += 1
                orders, depth = reorder(grid, best_route, best, old, left, width, wait, flight)
                attempt["completed_depth"] = depth
                for path in orders:
                    schedules, _ = retime(grid, path, old, wait, flight, best[:max(0, left - 3)])
                    for schedule in schedules:
                        submit(schedule, attempt)
                continue
            else:
                path, first, move = mutation(best_route, rng)
                prefix = best[:max(0, first - 3)]
                method = f"{move} with suffix timing DP"
            attempt.update(method=method, path=path, fixed_prefix_legs=len(prefix))
            stats["attempts"] += 1
            schedules, depth = retime(grid, path, old, wait, flight, prefix)
            attempt["completed_depth"] = depth
            for schedule in schedules:
                submit(schedule, attempt)
            if schedules:
                polish = Transfers(udp, deadline)
                refined = polish.polish(schedules[0], 0.)
                submit(refined, {**attempt, "method": f"{method} + continuous refinement"})
            if iteration % 5 == 4:
                print(f"{stats} | {time.monotonic()-started:.1f}s | best {best_days:.12f}", flush=True)
    except (BudgetExpired, KeyboardInterrupt):
        pass
    finally:
        for grid in grids.values():
            aggregates.update(grid.stats)
        summary = args.output_dir / f"easy_global_{run_id}_summary.json"
        write_json(summary, {"baseline": base, "best_official_days": best_days, "strategy": args.strategy,
                             "best_order_changed": best_route != original,
                             "best_changed_order_days": best_changed if math.isfinite(best_changed) else None,
                             "elapsed_seconds": time.monotonic() - started, "budget_seconds": args.seconds,
                             "counts": dict(stats), "distinct_feasible_orders": len(routes),
                             "transfer_counts": dict(aggregates), "attempts": report})
        print(f"{dict(stats)} | {dict(aggregates)} | best {best_days:.12f} | {summary}", flush=True)


def mutation(route, rng):
    n = len(route)
    left, right = sorted(rng.sample(range(n), 2))
    if rng.random() < 0.4:
        return route[:left] + route[left:right+1][::-1] + route[right+1:], left, "segment reversal"
    if rng.random() < 0.5:
        length = min(rng.randint(1, 5), n - right)
        path = route[:left] + route[right:right+length] + route[left:right] + route[right+length:]
        return path, left, "block relocation"
    middle = rng.randrange(left + 1, right + 1)
    return route[:left] + route[middle:right+1] + route[left:middle] + route[right+1:], left, "block exchange"


def structural_route(route, udp, rng, iteration):
    """Move entire low-shell groups and reverse sweeps between them."""
    elements = udp.tomato_orbital_parameters
    shell = (elements[:, 0] > np.median(elements[:, 0]) * .7)
    groups = [[route[0]]]
    for node in route[1:]:
        if shell[node] == shell[groups[-1][-1]]:
            groups[-1].append(node)
        else:
            groups.append([node])
    inner = [index for index, group in enumerate(groups) if len(group) > 1]
    if iteration % 5 == 2:
        index = rng.choice(inner)
        groups[index].reverse()
    elif iteration % 5 == 3:
        # Permute all nodes within the small groups, changing shell entries
        # and exits while keeping the large-shell science sweep intact.
        for group in groups:
            if not shell[group[0]]:
                rng.shuffle(group)
    else:
        index = rng.choice(inner)
        groups[index] = groups[index][1:] + groups[index][:1]
        if rng.randrange(2):
            groups.reverse()
    return tuple(node for group in groups for node in group)


def reorder(grid, route, schedule, old, left, width, wait, flight, displacement=None):
    """Propose whole-block permutations using beam-pruned subset labels."""
    nodes = route[left:left + width]
    suffix = route[left + width:]
    prefix = schedule[:max(0, left - 1)]
    normal = grid.udp.dv_threshold
    budget = grid.udp.n_dv_exception
    spent = sum(leg[4] > normal for leg in prefix)
    before = route[left - 1] if left else None
    day = arrival(prefix[-1]) if prefix else 0.
    states = {(0, before, spent): (day, ())}
    for depth in range(width):
        grid.check()
        merged = {}
        for (mask, last, used), (day, order) in states.items():
            indices = [index for index in range(width) if not mask & (1 << index)]
            if displacement:
                indices = indices[:displacement]
                if indices[0] + displacement < depth:
                    indices = indices[:1]
            for index in indices:
                node = nodes[index]
                if last is None:
                    options = (None,)
                else:
                    options = grid.edges(last, node, day, min(wait, 2.0), flight, used < budget)
                    old_leg = old.get((last, node))
                    if old_leg and old_leg[2] + 1e-6 >= day:
                        options += (old_leg,)
                for leg in options:
                    if leg is None and last is not None:
                        continue
                    count = used + int(leg is not None and leg[4] > normal)
                    finish = arrival(leg) if leg else day
                    if count > budget:
                        continue
                    key = (mask | (1 << index), node, count)
                    if key not in merged or finish < merged[key][0]:
                        merged[key] = (finish, order + (node,))
        if not merged:
            return [], depth
        selected = sorted(merged, key=lambda key: (merged[key][0] + 2.5 * key[2], key[0]))[:20]
        states = {key: merged[key] for key in selected}
    proposals = dict.fromkeys(route[:left] + label[1] + suffix for label in states.values() if label[1] != nodes)
    return list(proposals)[:8], width


if __name__ == "__main__":
    main()
