"""Exception-conditioned chain contraction and continuous chain scheduling for ESA Small."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from continuous_windows_easy import FILES, SEED, WindowSearch
from elite_easy import anchor_bank, scan_archive
from global_dp_easy import Grid, evaluator
from global_event_easy import Topology, geometry
from kttsp_esa_solver import validate
from pareto_easy import BudgetExpired, arrival, write_json


def edge(a, b):
    return min(a, b), max(a, b)


def forced_edges(normal, start, end, exceptions):
    """Propagate undirected residual degrees after fixing endpoints and exceptions."""
    n = len(normal)
    blocked = {edge(*pair) for pair in exceptions}
    demand = [2 - (i in (start, end)) for i in range(n)]
    for a, b in blocked:
        demand[a] -= 1
        demand[b] -= 1
    remaining = {edge(a, b) for a, b in zip(*np.nonzero(normal))} - blocked
    forced = set()
    while True:
        neighbors = [set() for _ in range(n)]
        for a, b in remaining:
            neighbors[a].add((a, b))
            neighbors[b].add((a, b))
        include, exclude = set(), set()
        for needed, incident in zip(demand, neighbors):
            if not 0 <= needed <= len(incident):
                raise ValueError("Inconsistent normal-degree demands")
            if needed == 0:
                exclude.update(incident)
            elif needed == len(incident):
                include.update(incident)
        if include & exclude:
            raise ValueError("Inconsistent forced edges")
        if not include and not exclude:
            return forced
        for a, b in include:
            demand[a] -= 1
            demand[b] -= 1
        forced.update(include)
        remaining.difference_update(include | exclude)


def blocks(path, exceptions, forced):
    """Contract maximal forced runs, retaining uncertain normal legs as connectors."""
    runs = []
    pending = [path[0]]
    for a, b in zip(path, path[1:]):
        if edge(a, b) in forced and (a, b) not in exceptions:
            pending.append(b)
            continue
        if len(pending) > 1:
            runs.append((tuple(pending), 0))
        runs.append(((a, b), int((a, b) in exceptions)))
        pending = [b]
    if len(pending) > 1:
        runs.append((tuple(pending), 0))
    return runs


class Search:
    # Reuse the Small-only scalar Lambert implementation. Every complete
    # candidate is checked separately by the unchanged official evaluator.
    cost = WindowSearch.cost

    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + args.seconds
        self.module = evaluator()
        self.udp = self.module.udp
        self.normal = self.udp.dv_threshold
        self.special = self.udp.dv_threshold_exception
        self.allowed, floor = geometry(self.udp, self.module.MU_MOON)
        self.shell = self.udp.tomato_orbital_parameters[:, 0] > 1e7
        self.stats = Counter()
        self.rng = random.Random(args.seed)
        self.archive, self.stats["validated_archive_files"] = scan_archive(self.udp)
        self.topology = Topology(self.udp, self.allowed, floor, self.archive)
        self.anchors = anchor_bank(self.archive)
        raw = json.loads(args.incumbent.read_text())
        self.path = tuple(raw["path"])
        self.baseline = validate(self.udp, self.path, raw["transfers"])
        if not self.baseline or not self.baseline["valid"]:
            raise ValueError("ESA rejected the incumbent")
        self.best = self.baseline["fitness"][0]
        self.incumbent = tuple((*l[:4], float(self.udp.compute_transfer(*l[:4])))
                               for l in raw["transfers"])
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        self.windows = defaultdict(set)
        self.allocations = {}
        self.seen = set()
        self.orders = set()
        self.history = []
        self.best_candidate = None
        self.best_novel = None
        self.reverse_grids = {}
        self.grid = Grid(self.module, .0625, .03125, self.deadline, directional=True)
        self.grid.horizon = min(self.best + 9., self.udp.max_time)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired

    def decompose(self, path, extra):
        forced = forced_edges(self.allowed[0], path[0], path[-1], extra)
        pieces = blocks(path, extra, forced)
        key = (path[0], path[-1], tuple(sorted(extra)))
        if key not in self.allocations:
            self.allocations[key] = {
                "endpoints": [path[0], path[-1]], "exceptions": sorted(extra),
                "shell_crossings": sorted((a, b) for a, b in extra if self.shell[a] != self.shell[b]),
                "forced_normal_edges": len(forced),
                "chain_lengths": [len(nodes) - 1 for nodes, _ in pieces],
                "contracted_chains": [nodes for nodes, _ in pieces if len(nodes) > 2]}
        return pieces

    def add(self, block, first, last=None):
        self.windows[block].add((tuple(first), tuple(first if last is None else last)))

    def make_profile(self, block, seed, latest, mode, radius):
        """Optimize the entire chain with a shared arrival deadline."""
        nodes, extra = block
        size = len(seed)
        xy = np.array([leg[2:4] for leg in seed]).ravel()
        cap = (self.special if extra else self.normal) - 1e-5
        lower = np.maximum(xy - np.tile([radius, radius * .55], size),
                           np.tile([0., self.udp.min_tof], size))
        upper = np.minimum(xy + np.tile([radius, radius * .55], size), self.udp.max_time)
        chrono = np.zeros((size, 2 * size))
        for i in range(size - 1):
            chrono[i, 2*i:2*i+3] = [-1., -1., 1.]
        chrono[-1, -2:] = [-1., -1.]
        rhs = np.r_[np.full(size - 1, -8e-7), -latest]
        weights = np.zeros(2 * size)
        if mode == "late":
            weights[0] = -1.
        else:
            weights[-2:] = 1.
        pairs = tuple(zip(nodes, nodes[1:]))

        def slack(i, d, f):
            return (cap - self.cost(*pairs[i], d, f)) / 100.

        def margin(x):
            return np.array([slack(i, *x[2*i:2*i+2]) for i in range(size)])

        def jacobian(x):
            jac = np.zeros((size, 2*size))
            for i in range(size):
                d, f = x[2*i:2*i+2]
                base = slack(i, d, f)
                h = 2e-7
                jac[i, 2*i:2*i+2] = [(slack(i, d+h, f)-base)/h, (slack(i, d, f+h)-base)/h]
            return jac

        self.stats["chain_continuous_solves"] += 1
        fit = minimize(lambda x: float(weights @ x), xy, jac=lambda x: weights, method="SLSQP",
                       bounds=Bounds(lower, upper), constraints=[
                           LinearConstraint(chrono, rhs, np.inf),
                           {"type": "ineq", "fun": margin, "jac": jacobian}],
                       options={"maxiter": 28, "ftol": 3e-9})
        if min(chrono @ fit.x - rhs) < -8e-8 or min(margin(fit.x)) < -3e-8:
            return None
        self.stats["successful_chain_solves"] += 1
        return tuple((*pair, float(d), float(f), self.cost(*pair, d, f))
                     for pair, (d, f) in zip(pairs, fit.x.reshape(-1, 2)))

    def interpolate(self, first, last, t):
        return tuple((a, b, d + t*(v[2]-d), f + t*(v[3]-f), dv)
                     for (a, b, d, f, dv), v in zip(first, last))

    def expand_window(self, block, seed, deadline):
        early = self.make_profile(block, seed, deadline, "early", .65)
        late = self.make_profile(block, seed, deadline, "late", .65)
        if early:
            self.add(block, early)
        if late:
            self.add(block, late)
        if not early or not late or abs(early[0][2] - late[0][2]) < 1e-5:
            return
        # A chord represents an entire correlated schedule, not a product
        # of independently interpolated leg choices.
        limit = (self.special if block[1] else self.normal) - 2e-7
        endpoint = late
        for _ in range(4):
            if all(self.cost(*leg[:4]) <= limit for t in (.125, .25, .5, .75, .875, 1.)
                   for leg in self.interpolate(early, endpoint, t)):
                self.add(block, early, endpoint)
                self.stats["continuous_chain_windows"] += 1
                return
            endpoint = self.interpolate(early, endpoint, .5)

    def import_archive(self):
        best_members = [min(group, key=lambda row: row[0]) for group in self.archive.values()]
        members = [self.incumbent, *(row[1] for row in sorted(best_members) if row[0] <= self.best+9.)]
        seeds = {}
        for legs in members:
            path = (legs[0][0], *(l[1] for l in legs))
            extra = {l[:2] for l in legs if l[4] > self.normal}
            pieces = self.decompose(path, extra)
            # Deadlines are actual following shell-crossing epochs where
            # available, tightened by the departure of the next block.
            offset = 0
            for block in pieces:
                end = offset + len(block[0]) - 1
                section = legs[offset:end]
                next_departure = legs[end][2] if end < len(legs) else arrival(legs[-1])
                self.add(block, section)
                phase = (block, round(section[0][2], 1), round(arrival(section[-1]), 1))
                seeds.setdefault(phase, (block, section, max(arrival(section[-1]), next_departure)))
                offset = end
        return list(seeds.values())

    def transitions(self, block, release):
        best = None
        for first, last in self.windows.get(block, ()):
            d, D = first[0][2], last[0][2]
            low, high = 0., 1.
            required = release - 8e-7
            if max(d, D) < required:
                continue
            if d < required:
                low = (required-d)/(D-d)
            elif D < required:
                high = (required-d)/(D-d)
            t = low if arrival(first[-1]) <= arrival(last[-1]) else high
            candidate = first if t == 0. else last if t == 1. else self.interpolate(first, last, t)
            finish = arrival(candidate[-1])
            if best is not None and finish >= best[0]:
                continue
            if 0. < t < 1.:
                limit = self.special + 1e-6 if block[1] else self.normal
                measured = tuple((*l[:4], self.cost(*l[:4])) for l in candidate)
                if any(l[4] > limit for l in measured):
                    self.stats["rejected_interpolations"] += 1
                    continue
                self.stats["interior_chain_choices"] += 1
                candidate = measured
            best = (finish, candidate)
        return best

    def bridge_chain(self, block, release, latest=None):
        """Initialize a new chain profile with an exact forward Lambert pass."""
        nodes, extra = block
        schedule = []
        latest = self.grid.horizon if latest is None else latest
        for a, b in zip(nodes, nodes[1:]):
            candidates = (*self.anchors.get((a,b), ()),
                          *self.grid.edges(a, b, release, 4., 8., bool(extra)))
            usable = [leg for leg in candidates if leg is not None and leg[2]+8e-7 >= release
                      and leg[4] <= (self.special + 1e-6 if extra else self.normal)
                      and arrival(leg) <= latest + 8e-7]
            if not usable:
                return None
            leg = min(usable, key=arrival)
            schedule.append(leg)
            release = arrival(leg)
        self.add(block, schedule)
        self.stats["new_chain_seeds"] += 1
        if len(nodes) > 2:
            self.expand_window(block, schedule, max(release, min(release + .25, latest)))
        return release, tuple(schedule)

    def backward_calendar(self, pieces, cap):
        """Propagate latest releases from the goal through the actual chain order."""
        if cap not in self.reverse_grids:
            self.reverse_grids[cap] = Grid(self.module, .0625, .03125, self.deadline,
                                           finish_day=cap, directional=True)
        grid = self.reverse_grids[cap]
        releases = {}
        deadline = cap
        for index in range(len(pieces)-1, -1, -1):
            nodes, extra = pieces[index]
            block = pieces[index]
            latest = deadline
            reverse = []
            for a, b in zip(nodes[-2::-1], nodes[:0:-1]):
                choices = [l for l in self.anchors.get((a, b), ())
                           if arrival(l) <= latest + 8e-7 and
                           l[4] <= (self.special + 1e-6 if extra else self.normal)]
                candidates = grid.edges(b, a, max(0., cap-latest), 4., 8., bool(extra))
                for leg in candidates:
                    if leg is None:
                        continue
                    _, _, d, f, _ = leg
                    dep = cap-d-f
                    if dep < 0. or dep+f > latest + 8e-7:
                        continue
                    dv = self.cost(a, b, dep, f)
                    if dv <= (self.special + 1e-6 if extra else self.normal):
                        choices.append((a,b,dep,f,dv))
                if not choices:
                    return releases
                leg = max(choices, key=lambda l: l[2])
                latest = leg[2]
                reverse.append(leg)
            self.add(block, reverse[::-1])
            self.stats["backward_chain_profiles"] += 1
            releases[index] = deadline
            deadline = latest
        return releases

    def schedule_topology(self, pieces, calendar=None):
        releases = 0.
        legs = ()
        for pos, block in enumerate(pieces):
            self.check()
            choice = self.transitions(block, releases)
            latest = calendar.get(pos) if calendar else None
            # Try a fresh continuous chain seed when cached profiles cannot
            # clear the next connector or are separated by a large phase gap.
            if choice is None or choice[1][0][2] > releases + .8 or (
                    latest is not None and choice[0] > latest + 8e-7):
                alternative = self.bridge_chain(block, releases, latest)
                if alternative and (choice is None or alternative[0] < choice[0]):
                    choice = alternative
            if choice is None:
                return None, pos
            releases, selected = choice
            legs += selected
        return legs, len(pieces)

    def schedule_global(self, cap):
        """Resource-constrained DP over chain order and continuous chain profiles."""
        n = self.udp.n_tomatoes
        outgoing = defaultdict(list)
        for (nodes, extra), windows in sorted(self.windows.items()):
            added = sum(1 << i for i in nodes[1:])
            earliest = min(arrival(s[-1]) for window in windows for s in window)
            outgoing[nodes[0]].append(((nodes, extra), added, earliest))
        layers = [dict() for _ in range(n+1)]
        for node in outgoing:
            layers[1][(1 << node, node, 0, 0)] = (0., ())
        complete = []
        for count in range(1, n+1):
            self.check()
            # Time per visit is meaningful across compact chains only at the
            # same layer. Beam size bounds memory and compute.
            layer = sorted(layers[count].items(), key=lambda row: row[1][0])[:self.args.beam]
            self.stats["global_chain_labels"] += len(layer)
            for (mask, node, used, crosses), (day, legs) in layer:
                if count == n:
                    complete.append(legs)
                    continue
                if self.udp.n_dv_exception - used < max(0, 4-crosses):
                    continue
                for block, added, earliest in outgoing[node]:
                    nodes, extra = block
                    if mask & added or used+extra > self.udp.n_dv_exception:
                        continue
                    remaining = n-count-len(nodes)+1
                    if earliest + remaining*self.udp.min_tof > cap + 1e-5:
                        continue
                    candidate = self.transitions(block, day)
                    if not candidate:
                        continue
                    finish, segment = candidate
                    if finish + remaining*self.udp.min_tof > cap + 1e-5:
                        continue
                    new_crosses = crosses + int(self.shell[node] != self.shell[nodes[-1]])
                    key = mask | added, nodes[-1], used+extra, new_crosses
                    target = layers[n-remaining]
                    if key not in target or finish < target[key][0]:
                        target[key] = (finish, legs + segment)
            layers[count].clear()
        return complete

    def submit(self, legs, mode):
        signature = tuple(l[:4] for l in legs)
        if signature in self.seen:
            return
        self.seen.add(signature)
        path = (legs[0][0], *(l[1] for l in legs))
        result = validate(self.udp, path, legs)
        self.stats["esa_checks"] += 1
        if not result or not result["valid"]:
            self.stats["esa_rejected"] += 1
            return
        self.stats["esa_feasible"] += 1
        self.orders.add(path)
        days = result["fitness"][0]
        info = {"path": path, "transfers": legs, "validation": result, "mode": mode,
                "flight_days": sum(l[3] for l in legs), "seed": self.args.seed,
                "incumbent": str(self.args.incumbent)}
        if self.best_candidate is None or days < self.best_candidate["validation"]["fitness"][0]:
            self.best_candidate = info
        if path not in self.archive and (self.best_novel is None or
                                         days < self.best_novel["validation"]["fitness"][0]):
            self.best_novel = info
        if days < self.best:
            self.best = days
            self.stats["records"] += 1
            write_json(FILES / f"easy_contracted_{self.run_id}_{self.stats['records']:04}.json", info)
            print(f"ESA record: {days:.14f} days", flush=True)

    def run(self):
        try:
            seeds = self.import_archive()
            print(f"Contracted {len(self.allocations)} archived allocations; {len(seeds)} seed profiles",
                  flush=True)
            # Start from the incumbent's long chains and then alternate the
            # deadline of each phase rather than perturbing isolated legs.
            jobs = [(block, legs, deadline) for block, legs, deadline in seeds if len(block[0]) > 2]
            jobs.sort(key=lambda row: -(len(row[0][0])-1))
            for k, (block, legs, deadline) in enumerate(jobs[:self.args.profiles]):
                for shift in (0., -.16) if k < self.args.profiles // 2 else (0.,):
                    self.expand_window(block, legs, max(legs[0][2], deadline + shift))
            for legs in self.schedule_global(self.best + 2.):
                self.submit(legs, "contracted global profile DP")
            print(f"Chain diagnostics: {dict(self.stats)}", flush=True)
            reference = set(zip(self.path, self.path[1:]))
            for i in range(self.args.topologies):
                self.check()
                # Geometry-only proposal. No individual transfer time
                # variables are used in the topology MIP.
                fit = self.topology.propose(self.rng, min(self.deadline, time.monotonic()+4.),
                                            reference, 4 if i % 2 else 2, .7+.35*(i % 3))
                if fit is None:
                    self.stats["topology_no_route"] += 1
                    continue
                path, exceptions, score = fit
                self.stats["static_topologies"] += 1
                pieces = self.decompose(path, set(exceptions))
                legs, reached = self.schedule_topology(pieces)
                backward_depth = 0
                if self.args.backward and legs is None:
                    calendar = self.backward_calendar(pieces, self.baseline["fitness"][0] +
                                                       (1.5 if i % 2 else 5.))
                    backward_depth = len(calendar)
                    if backward_depth:
                        legs, reached = self.schedule_topology(pieces, calendar)
                self.history.append({"path": path, "allocated_exceptions": exceptions,
                                     "blocks_reached": reached, "backward_blocks": backward_depth, "blocks": len(pieces),
                                     "static_flight_proxy": score})
                if legs:
                    self.submit(legs, "new exception-conditioned topology")
            for legs in self.schedule_global(self.best + 4.):
                self.submit(legs, "expanded contracted global profile DP")
        except BudgetExpired:
            self.stats["deadline"] += 1
        finally:
            report = {"best_days": self.best, "baseline": self.baseline, "counts": dict(self.stats),
                      "elapsed_seconds": time.monotonic()-self.started,
                      "distinct_feasible_orders": len(self.orders), "allocations": list(self.allocations.values()),
                      "new_topologies": self.history, "best_generated_candidate": self.best_candidate,
                      "best_novel_candidate": self.best_novel,
                      "config": {**vars(self.args), "incumbent": str(self.args.incumbent)}}
            file = FILES / f"easy_contracted_{self.run_id}_summary.json"
            write_json(file, report)
            print(f"{dict(self.stats)} | {file}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--incumbent", type=Path, default=SEED)
    p.add_argument("--seconds", type=float, default=130.)
    p.add_argument("--topologies", type=int, default=12)
    p.add_argument("--profiles", type=int, default=28)
    p.add_argument("--beam", type=int, default=2000)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--seed", type=int, default=731)
    Search(p.parse_args()).run()
