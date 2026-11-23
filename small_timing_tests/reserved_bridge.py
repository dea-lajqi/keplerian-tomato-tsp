"""Bounded Small benchmark for a join with an exception held in reserve."""
import argparse
from collections import Counter, defaultdict
from itertools import pairwise
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_windows_easy import FILES
from focus_easy import refine_joint
from global_dp_easy import Grid, evaluator
from global_event_easy import fit_transfer_window, geometry
from pareto_easy import BudgetExpired, arrival
from small_timing_tests.controlled import Results, corpus
from small_v2 import atomic_json


class Search:
    def __init__(self, seconds, goal, phases, lattice_variants):
        self.start = time.monotonic()
        self.stop = self.start + seconds
        self.goal = goal
        self.module = evaluator()
        self.udp = self.module.udp
        self.limit = self.udp.dv_threshold
        self.cap = self.udp.dv_threshold_exception + 1e-6
        self.side_budget = self.udp.n_dv_exception - 1
        self.normal, self.allowed = geometry(self.udp, self.module.MU_MOON)[0]
        self.seed, parents, learned = corpus(self.udp)
        self.parents = [tuple(row["path"]) for row in parents]
        self.results = Results(self.udp)
        self.counts = Counter()
        self.trials = []
        self.closed = {}
        self.lattice_variants = lattice_variants
        self.rng = random.Random(13497)
        self.events = defaultdict(set)
        atlas = FILES / "small_v4/diagnostic/window_e3e82fc984_checkpoint.json"
        for tail in json.loads(atlas.read_text())["windows"]:
            if arrival(tail[-1]) <= goal:
                learned.extend(map(tuple, tail))
        for a, b, d, f, _ in set(learned):
            if d < 0 or d + f > goal or f < self.udp.min_tof:
                continue
            cost = float(self.udp.compute_transfer(a, b, d, f))
            if cost <= self.cap:
                self.events[a, b].add((a, b, d, f, cost))
        self.grids = []
        for step, phase in phases:
            forward = Grid(self.module, step, phase, self.stop, directional=True)
            backward = Grid(self.module, step, phase, self.stop,
                            finish_day=goal, directional=True)
            backward.z_speed = -backward.z_speed
            self.grids.append((forward, backward))

    def orders(self):
        """Geometry-screen relocations on either side of every archived exception."""
        def changes(path):
            first = tuple(pairwise(path))
            critical = [i for i, (a, b) in enumerate(first)
                        if not self.normal[a, b]]
            for i in sorted({k for j in critical for k in range(max(0, j-1), min(49, j+3))}):
                rest = path[:i] + path[i+1:]
                for j in range(49):
                    if j != i:
                        candidate = rest[:j] + (path[i],) + rest[j:]
                        edges = tuple(pairwise(candidate))
                        if all(self.allowed[a, b] for a, b in edges):
                            mandatory = sum(not self.normal[a, b] for a, b in edges)
                            if mandatory <= self.udp.n_dv_exception:
                                new_edges = set(edges) - set(first)
                                missing = sum(not self.events[edge] for edge in new_edges)
                                yield (mandatory, missing, candidate)
        variants = [v for parent in self.parents for v in changes(parent)]
        ordered = sorted(set(variants))
        chosen = self.parents + [row[2] for row in ordered[:300]]
        return list(dict.fromkeys(chosen)), list(dict.fromkeys(self.parents + [
            row[2] for row in ordered[:self.lattice_variants]]))

    def labels(self, path, reverse=False, grid=None):
        """One earliest label per exact exception count, computed at every cut."""
        order = path[::-1] if reverse else path
        frontier = {0: (0., ())}
        result = {order[0]: frontier}
        remaining = tuple(pairwise(order))
        for a, b in remaining:
            if time.monotonic() >= self.stop:
                raise BudgetExpired
            edge = (b, a) if reverse else (a, b)
            next_labels = {}
            for used, (clock, schedule) in frontier.items():
                choices = list(self.events[edge])
                if grid is not None:
                    for leg in grid.edges(a, b, clock, 5., 9.5, used < self.side_budget):
                        if leg is None:
                            continue
                        _, _, dep, tof, _ = leg
                        d = self.goal - dep - tof if reverse else dep
                        if d >= 0 and d + tof <= self.goal + 1e-9:
                            dv = float(self.udp.compute_transfer(*edge, d, tof))
                            if dv <= self.cap:
                                choices.append((*edge, d, tof, dv))
                for leg in choices:
                    dep = self.goal - arrival(leg) if reverse else leg[2]
                    finish = dep + leg[3]
                    spend = used + int(leg[4] > self.limit)
                    if spend > self.side_budget or dep < clock - 1e-8:
                        continue
                    if spend not in next_labels or finish < next_labels[spend][0]:
                        next_labels[spend] = (finish, (*schedule, leg))
            if not next_labels:
                break
            frontier = next_labels
            result[b] = frontier
        return result

    def stage(self, paths, grid_pair=(None, None), max_trials=30):
        forward, backward = defaultdict(dict), defaultdict(dict)
        for path in paths:
            for rev, table, grid in ((False, forward, grid_pair[0]), (True, backward, grid_pair[1])):
                labels = self.labels(path, rev, grid)
                mask = 0
                for x in (path[::-1] if rev else path):
                    mask |= 1 << x
                    if x not in labels:
                        break
                    if not 3 <= mask.bit_count() <= 46:
                        continue
                    for spend, (clock, schedule) in labels[x].items():
                        key = x, spend
                        if key not in table[mask] or clock < table[mask][key][0]:
                            table[mask][key] = (clock, schedule)
        full = (1 << 49) - 1
        self.counts["forward_coverage_sets"] += len(forward)
        self.counts["backward_coverage_sets"] += len(backward)
        proposals = []
        for mask, prefixes in forward.items():
            for (a, used), (clock, pre) in prefixes.items():
                for (b, spent), (reverse_clock, post) in backward.get(full ^ mask, {}).items():
                    self.counts["complementary_label_pairs"] += 1
                    if used + spent > self.side_budget:
                        self.counts["over_reserved_budget"] += 1
                    elif not self.allowed[a, b]:
                        self.counts["geometry_rejections"] += 1
                    else:
                        due = self.goal - reverse_clock
                        span = due - clock - self.udp.min_tof
                        if span <= 0:
                            self.counts["closed_time_windows"] += 1
                            key = a, b, used, spent
                            previous = self.closed.get(key)
                            if previous is None or span > previous["slack_after_min_tof"]:
                                self.closed[key] = {
                                    "edge": (a, b), "prefix_exceptions": used,
                                    "suffix_exceptions": spent, "release": clock,
                                    "due": due, "slack_after_min_tof": span,
                                }
                        else:
                            self.counts["open_reserved_windows"] += 1
                            proposals.append((a, b, used, spent, clock, due, pre, post))
        proposals.sort(key=lambda p: (self.normal[p[0], p[1]], -(p[5] - p[4])))
        # Avoid exhausting the test budget on repeated epochs for one edge.
        tested_edges = Counter()
        for a, b, used, spent, release, due, pre, post in proposals:
            if max_trials <= 0:
                break
            if tested_edges[a, b] >= 3:
                continue
            key = a, b, round(release, 7), round(due, 7)
            if key in self.tested:
                continue
            self.tested.add(key)
            tested_edges[a, b] += 1
            max_trials -= 1
            fits, minimum = fit_transfer_window(self.udp, a, b, release, due, self.cap,
                                                self.stop, self.rng)
            trial = {"edge": (a, b), "release": release, "due": due,
                     "prefix_exceptions": used, "suffix_exceptions": spent,
                     "minimum_sampled_dv": minimum}
            self.trials.append(trial)
            self.counts["continuous_bridge_checks"] += 1
            if not fits:
                self.counts["bridge_dv_search_failed"] += 1
            for bridge in fits:
                legs = (*pre, bridge, *post[::-1])
                path = (legs[0][0], *(leg[1] for leg in legs))
                exception_count = sum(leg[4] > self.limit for leg in legs)
                assert len(set(path)) == len(path) == 49 and exception_count <= 5
                trial["complete_exceptions"] = exception_count
                trial["bridge_dv"] = bridge[4]
                self.results.add(path, legs)

    def run(self):
        self.tested = set()
        paths, lattice_paths = self.orders()
        self.counts.update(event_orders=len(paths), lattice_orders=len(lattice_paths))
        try:
            self.stage(paths)
            for grids in self.grids:
                self.stage(lattice_paths, grids)
        except BudgetExpired:
            self.counts["time_limit_reached"] += 1
        control = Results(self.udp)
        control.add(self.seed["path"], self.seed["transfers"])
        return {"deadline": self.goal, "wall_seconds": time.monotonic()-self.start,
                "counts": dict(self.counts), "trials": self.trials,
                "closest_closed_windows": sorted(self.closed.values(),
                    key=lambda row: -row["slack_after_min_tof"])[:12],
                "baseline": control.report()["independent"], "complete_tours": self.results.report()}


def refine_control(source, seconds):
    module = evaluator()
    row = json.loads(source.read_text())["complete_tours"]["best_schedule"]
    schedule = tuple(map(tuple, row["transfers"]))
    path = tuple(row["path"])
    result = Results(module.udp)
    result.add(path, schedule)
    start = time.monotonic()
    attempts = 0
    try:
        for first, shift in ((37, -.45), (32, -.95), (25, -1.25)):
            refine_joint(module.udp, schedule, start + seconds,
                         lambda legs: result.add(path, legs), 0.,
                         random.Random(5100 + first), False,
                         coherent_shift=(first, len(schedule), shift))
            attempts += 1
    except BudgetExpired:
        pass
    return {"wall_seconds": time.monotonic()-start, "completed_attempts": attempts,
            "complete_tours": result.report()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=45.)
    parser.add_argument("--deadline", type=float, default=111.)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--denser", action="store_true")
    parser.add_argument("--refine-control", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    phases = [(.09375, .017), (.078125, .037)]
    if args.denser:
        phases.extend([(.0625, .011), (.046875, .022)])
    if args.refine_control:
        results = refine_control(args.refine_control, args.seconds)
    else:
        results = Search(args.seconds, args.deadline, phases, 64 if args.denser else 28).run()
    atomic_json(args.output, results)
    print(results["wall_seconds"], results.get("counts", results.get("completed_attempts")),
          results["complete_tours"]["independent"])
