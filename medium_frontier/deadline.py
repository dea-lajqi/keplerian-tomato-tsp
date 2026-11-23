"""Reverse terminal deadlines and jointly timed overlapping high-shell windows."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import random
import time

from global_event_easy import fit_transfer_window

from medium_coupled.checkpoint import atomic_json
from medium_coupled.exact_terminal import ExactTerminal
from medium_coupled.meet import SideGrid
from medium_coupled.run import Paused, solve
from medium_coupled.terminal_atlas import pareto
from medium_overnight.validation import solution_payload, verify
from pareto_easy import BudgetExpired, arrival


class Deadline(ExactTerminal):
    def __init__(self, args):
        super().__init__(args)
        self.rows = defaultdict(set)
        for file in (args.atlas, *args.events):
            for key, events in json.loads(file.read_text())["rows"].items():
                self.rows[tuple(map(int, key.split(",")))].update(map(tuple, events))
        for pair, rows in self.anchors.items():
            self.rows[pair].update(rows)
        self.index = {}
        self.learned = defaultdict(set)
        self.learned_file = args.directory / "learned_events.json"
        if self.learned_file.exists():
            for key, rows in json.loads(self.learned_file.read_text())["rows"].items():
                edge = tuple(map(int, key.split(",")))
                self.learned[edge].update(map(tuple, rows))
                self.rows[edge].update(self.learned[edge])
        self.sampled = set()
        self.archive = args.directory / "feasible"
        self.crossing = self.shells[-1]

    def remember(self, row):
        key = row[:2]
        if row not in self.rows[key]:
            self.rows[key].add(row)
            self.learned[key].add(row)
            self.index.pop(key, None)

    def edge_index(self, key):
        if key not in self.index:
            classes = [pareto(r for r in self.rows[key] if int(r[-1] > self.normal) == cls)
                       for cls in (0, 1)]
            self.index[key] = [(tuple(r[2] for r in rows), tuple(map(arrival, rows)), rows)
                               for rows in classes]
        return self.index[key]

    def save_events(self):
        atomic_json(self.learned_file, {"rows": {f"{a},{b}": sorted(rows)
                    for (a, b), rows in self.learned.items()}})

    def options(self, a, b, deadline, exception, grid, sample):
        key = a, b
        if sample:
            wait = grid.origin - deadline
            tick = math.ceil((wait - grid.phase - 1e-10) / grid.step)
            query = key, grid.step, grid.phase, tick, exception
            if query not in self.sampled:
                self.sampled.add(query)
                for row in grid.edges(b, a, wait, 3.6, 8., exception):
                    if row is None:
                        continue
                    tof = row[3]
                    dep = grid.origin - row[2] - tof
                    dv = float(self.udp.compute_transfer(a, b, dep, tof))
                    self.count["event_replays"] += 1
                    if dv <= self.cap:
                        self.remember((a, b, dep, tof, dv))
        for _, arrivals, rows in self.edge_index(key)[:1 + exception]:
            idx = bisect_right(arrivals, deadline + 1e-9) - 1
            if idx >= 0:
                yield rows[idx]

    def keep(self, legs, tag):
        solution = self.build(legs)
        if not self.submit(solution, "deadline_joint", tag):
            return
        file = self.archive / (sha256(bytes(solution["path"])).hexdigest() + ".json")
        fitness = verify(self.udp, solution)
        if not file.exists() or fitness[0] < json.loads(file.read_text())["arrival_day"]:
            payload = solution_payload(solution, fitness, udp=self.udp,
                                       provenance={"method": "deadline_joint", "trial": tag})
            atomic_json(file, {"solution": solution, **payload})
        if fitness[0] < self.state["best"] + .05:
            try:
                solve(self.udp, solution, max(0, tag[1] - 2), 2.4, 0.,
                      min(self.stop, time.monotonic() + 6.),
                      lambda sol: self.submit(sol, "deadline_joint_polish", tag), self.count)
            except Paused:
                pass

    def backward(self, day, floor):
        group = tuple(self.path[self.crossing + 1:])
        if len(group) != 20:
            raise ValueError("Expected a complete 20-tomato terminal group")
        rows = {key: pareto(r for r in self.rows[key] if r[-1] <= self.normal and r[2] >= floor)
                for key in itertools.permutations(group, 2)}
        metrics, labels = self.execute(group, dict.fromkeys(group, -day), rows, 0., True)
        starts = {route[-1]: (-clock, self.reconstruct_reverse(route, rows, day))
                  for clock, route in labels}
        atomic_json(self.args.directory / f"terminal_{day}.json",
                    {"deadline": day, "floor": floor, **metrics,
                     "windows": {str(k): {"latest": t, "tail": legs}
                                 for k, (t, legs) in starts.items()}})
        return starts

    def forward_options(self, a, b, release, exception, grids, sample):
        key = a, b
        options = set()
        if sample:
            for grid in grids:
                before = len(options)
                for row in grid.edges(a, b, release - grid.origin, 4.6, 8.5, exception):
                    if row is None:
                        continue
                    dep, tof = grid.origin + row[2], row[3]
                    dv = float(self.udp.compute_transfer(a, b, dep, tof))
                    self.count["forward_replays"] += 1
                    if dep >= release - 1e-9 and dv <= self.cap:
                        leg = (a, b, dep, tof, dv)
                        options.add(leg)
                        self.remember(leg)
                if len(options) > before:
                    break
        for departures, _, rows in self.edge_index(key)[:1 + exception]:
            idx = bisect_left(departures, release - 1e-9)
            if idx < len(rows):
                options.add(rows[idx])
        return options

    def retain(self, values, width, priority):
        """Keep separate quota for partial paths with different ΔV resources."""
        ranked = sorted(values.items(), key=priority)
        if len(ranked) <= width:
            return dict(ranked)
        resources = {key[2] for key, _ in ranked}
        quota = max(1, width // (2 * len(resources)))
        counts = defaultdict(int)
        kept = {}
        for key, value in ranked:
            if counts[key[2]] < quota:
                kept[key] = value
                counts[key[2]] += 1
        for key, value in ranked:
            if len(kept) >= width:
                break
            kept[key] = value
        return dict(sorted(kept.items(), key=priority))

    def bridge(self, day, left, band, beam, step, layers, boundaries, prefix, budget, stop):
        """Join exact complementary coverage in the interior of the high passage."""
        floor = arrival(prefix[-1])
        best_gap = None
        matched = valid = 0
        refinements = {}
        near = {}
        refine_seconds = (19. if self.args.repair_schedule else 7.) if self.args.refine else 0.
        grids = [SideGrid(self.udp, floor, day, step, p, stop-refine_seconds, False)
                 for p in (.004, .024)]
        frontier = {(0, self.path[left], 0): (floor, 0., prefix)}
        def forward_priority(item):
            return item[1][0] + self.args.flight_weight*item[1][1]
        j = left
        try:
            for j in range(left, max(layers)):
                grids[0].check()
                ranking = sorted(frontier.items(), key=forward_priority)
                right = self.crossing
                candidates = defaultdict(list)
                for (mask, node, used), (due, _, tail) in layers.get(j + 1, {}).items():
                    candidates[mask].append((node, used, due, tail))
                next_frontier = {}
                for rank, ((mask, a, used), (clock, flight, chain)) in enumerate(ranking):
                    for b, spend, due, tail in candidates.get(boundaries[-1] ^ mask, ()):
                        remaining = budget - used - spend
                        if remaining < 0 or not self.allowed[a, b]:
                            continue
                        matched += 1
                        options = self.forward_options(a, b, clock, remaining > 0,
                                                       grids, rank < self.args.sample_labels)
                        if self.args.refine and clock + self.udp.min_tof < due:
                            key = a, b, round(clock, 2), round(due, 2), remaining > 0
                            gap0 = min((arrival(leg)-due for leg in options), default=3.)
                            if gap0 > 1e-9 and (len(refinements) < 20 or
                                               gap0 < max(row[0] for row in refinements.values())):
                                refinements[key] = gap0, clock, due, chain, tail
                                if len(refinements) > 20:
                                    del refinements[max(refinements, key=lambda k: refinements[k][0])]
                        for leg in options:
                            cost = int(leg[-1] > self.normal)
                            if cost > remaining:
                                continue
                            gap = arrival(leg) - due
                            if best_gap is None or gap < best_gap["days"]:
                                best_gap = {"days": gap, "edge": [a, b], "index": j,
                                            "arrival": arrival(leg), "latest": due,
                                            "exceptions": used + spend + cost}
                            if self.args.repair_schedule and 1e-9 < gap <= 2.6:
                                # Distinct full orders, each with a single
                                # chronology deficit at its bridge.
                                route = (*[l[1] for l in chain], b, *[l[1] for l in tail])
                                value = (gap, j, (*chain, leg, *tail))
                                if route not in near or gap < near[route][0]:
                                    near[route] = value
                                if len(near) > 12:
                                    del near[max(near, key=lambda key: near[key][0])]
                            if gap <= 1e-9 and valid < 12:
                                self.keep((*chain, leg, *tail), (day, left, band, step))
                                valid += 1
                    offset = j + 1 - left
                    lo, hi = max(0, offset-band-1), min(right-left, offset+band)
                    missing = boundaries[lo] & ~mask
                    if missing.bit_count() > 1:
                        continue
                    bits = missing or ((boundaries[hi] ^ boundaries[lo]) & ~mask)
                    while bits:
                        bit = bits & -bits
                        bits ^= bit
                        b = bit.bit_length() - 1
                        if not self.allowed[a,b] or (used == budget-1 and not self.normal_edges[a,b]):
                            continue
                        options = self.forward_options(a, b, clock, used < budget-1, grids,
                                                       rank < self.args.sample_labels)
                        for leg in options:
                            e = used + int(leg[-1] > self.normal)
                            finish = arrival(leg)
                            key = mask | bit, b, e
                            if e < budget and (key not in next_frontier or finish < next_frontier[key][0]):
                                next_frontier[key] = finish, flight + leg[3], (*chain, leg)
                frontier = (self.retain(next_frontier, beam, forward_priority) if self.args.refine
                            else dict(sorted(next_frontier.items(),key=forward_priority)[:beam]))
                if not frontier:
                    break
        except BudgetExpired:
            self.count["bridge_time_caps"] += 1
        finally:
            for grid in grids:
                self.count.update(grid.stats)
        repairs = []
        for gap, index, chain in sorted(near.values())[:5]:
            if time.monotonic() >= stop - 2.:
                break
            candidate = self.build(chain)
            key = (day, left, band, step, index, gap)
            name = sha256(bytes(candidate["path"])).hexdigest()[:16]
            path = self.args.directory / "near" / f"{day}_{left}_{name}.json"
            atomic_json(path, {"solution": candidate, "chronology_gap": gap, "trial": key})
            before = self.state["best"]
            try:
                result = solve(self.udp, candidate, max(left, index-14), 4.0, 0.,
                               min(stop-1.,time.monotonic()+3.8),
                               lambda sol: self.submit(sol, "deadline_joint_near_repair", key),
                               self.count, cap_flags=[l[-1] > self.normal for l in chain])
            except Paused:
                result = {"time_cap": True}
            repairs.append({"gap": gap, "index": index,
                            "gain": before-self.state["best"], **result})
        continuous = 0
        for (a, b, _, _, exception), (_, clock, due, chain, tail) in sorted(
                refinements.items(), key=lambda item: item[1][0]):
            if time.monotonic() >= stop or valid >= 12:
                break
            continuous += 1
            try:
                fits, _ = fit_transfer_window(self.udp, a, b, clock, due,
                    self.cap if exception else self.normal,
                    min(stop, time.monotonic() + 1.2), random.Random(18000+a*181+b))
            except BudgetExpired:
                continue
            for leg in fits:
                if leg[2] >= clock-1e-9 and arrival(leg) <= due+1e-9:
                    self.remember(leg)
                    self.keep((*chain, leg, *tail), (day, left, band, step))
                    valid += 1
        return {"matched": matched, "connected": valid, "closest": best_gap,
                "continuous_repairs": continuous, "schedule_repairs": repairs,
                "forward_index": j}

    def joint(self, day, starts, left, band, beam, step, trial_stop):
        """Search one continuous coverage frontier across every internal block seam."""
        path = self.path
        right = self.crossing
        boundaries = [0]
        for node in path[left + 1:right + 1]:
            boundaries.append(boundaries[-1] | (1 << node))
        prefix = tuple(self.legs[:left])
        budget = self.udp.n_dv_exception - sum(l[-1] > self.normal for l in prefix)
        floor = arrival(prefix[-1])
        reverse_stop = (trial_stop + time.monotonic()) / 2 if self.args.bridge else trial_stop
        grids = [SideGrid(self.udp, floor, day, step, p, reverse_stop, True)
                 for p in (.003, .019)]
        frontier = {(0, b, 0): (due, sum(l[3] for l in tail), tail)
                    for b, (due, tail) in starts.items()}
        depths, crossing_latest, layers = [], {}, {}
        def reverse_priority(item):
            return -item[1][0] + self.args.flight_weight*item[1][1], item[1][1]
        try:
            for j in range(right, left - 1, -1):
                grids[0].check()
                depth = j - left
                lo, hi = max(0, depth - band - 1), min(right-left, depth + band)
                must = boundaries[-1] ^ boundaries[hi]
                eligible = boundaries[hi] ^ boundaries[lo]
                new = {}
                ranking = sorted(frontier.items(), key=reverse_priority)
                for rank, ((mask, b, used), (due, flight, tail)) in enumerate(ranking):
                    if must & ~mask and j != right:
                        continue
                    bits = eligible & ~mask if j != left else 1 << path[left]
                    while bits:
                        bit = bits & -bits
                        bits ^= bit
                        a = bit.bit_length() - 1
                        if not self.allowed[a, b] or (used == budget and not self.normal_edges[a,b]):
                            continue
                        # Reserve all exceptional transitions still needed by
                        # the unvisited seed group sequence inside this window.
                        reserved = sum(k < j and k >= left for k in self.shells)
                        extra = used + reserved < budget
                        sample = j == right or rank < self.args.sample_labels
                        options = set(self.options(a, b, due, extra, grids[0], sample))
                        if sample and (not options or j == right):
                            options.update(self.options(a, b, due, extra, grids[1], True))
                        for leg in options:
                            spend = used + int(leg[-1] > self.normal)
                            ready = leg[2]
                            if j == right and (a not in crossing_latest or ready > crossing_latest[a]):
                                crossing_latest[a] = ready
                            if ready < floor - 1e-9:
                                continue
                            key = mask | bit, a, spend
                            value = ready, flight + leg[3], (leg, *tail)
                            previous = new.get(key)
                            if previous is None or (ready, -value[1]) > (previous[0], -previous[1]):
                                new[key] = value
                depths.append({"index": j, "labels": len(new),
                               "latest": max((v[0] for v in new.values()), default=None)})
                if not new:
                    break
                frontier = (self.retain(new, beam, reverse_priority) if self.args.refine
                            else dict(sorted(new.items(),key=reverse_priority)[:beam]))
                if self.args.bridge and j > left:
                    layers[j] = frontier
                if j == left:
                    for key, (_, _, tail) in list(frontier.items())[:8]:
                        if key[0] == boundaries[-1] | (1 << path[left]):
                            self.keep((*prefix, *tail), (day, left, band, step))
        except BudgetExpired:
            self.count["trial_time_caps"] += 1
        finally:
            for grid in grids:
                self.count.update(grid.stats)
        bridge = self.bridge(day, left, band, beam, step, layers, boundaries, prefix, budget, trial_stop) if layers else None
        return {"bridge": bridge, "day": day, "cut": left, "band": band, "beam": beam, "step": step,
                "crossing_latest": crossing_latest, "depths": depths}

    def run(self):
        for day in self.args.days:
            if time.monotonic() >= self.stop:
                break
            try:
                starts = self.backward(day, self.args.floor)
            except TimeoutError:
                break
            for cut, band, beam, step in self.args.trials:
                cut, band, beam = int(cut), int(band), int(beam)
                key = [day, cut, band, beam, step]
                if not starts or time.monotonic() >= self.stop or any(
                        t["key"] == key for t in self.state["trials"]):
                    continue
                now = time.monotonic()
                outcome = self.joint(day, starts, cut, band, beam, step,
                                     min(self.stop, now + self.args.trial_seconds))
                self.state["trials"].append({"key": key, "seconds": time.monotonic()-now,
                                              **outcome})
                self.save_events()
                self.save()
                print(key, len(starts), outcome["crossing_latest"], outcome["depths"][-2:],
                      self.state["best"], flush=True)
        self.state["wall_seconds"] = time.monotonic() - self.started
        self.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "seed-record", "atlas"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--events", type=Path, nargs="*", default=())
    parser.add_argument("--anchors", type=Path, nargs="*", default=())
    parser.add_argument("--days", type=float, nargs="+", default=(180., 181., 181.7))
    parser.add_argument("--floor", type=float, default=175.)
    parser.add_argument("--seconds", type=float, default=480.)
    parser.add_argument("--trial-seconds", type=float, default=42.)
    parser.add_argument("--bridge", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--repair-schedule", action="store_true")
    parser.add_argument("--flight-weight", type=float, default=0.)
    parser.add_argument("--sample-labels", type=int, default=6)
    parser.add_argument("--trials", type=float, nargs=4, action="append")
    parser.set_defaults(mode="deadline", beam=None, phase=None, goal=180.,
                        baseline_record=None)
    args = parser.parse_args()
    if args.repair_schedule:
        args.refine = True
    if args.trials is None:
        args.trials = [(139, 4, 900, .056), (129, 6, 1700, .047), (111, 6, 1800, .065)]
    Deadline(args).run()
