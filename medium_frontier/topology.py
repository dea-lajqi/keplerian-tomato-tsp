"""Enumerate measured-spine shell topologies with a hard geometry resource bound."""
import argparse
from collections import Counter, defaultdict
from itertools import permutations, product, pairwise
from pathlib import Path
import time

from medium_coupled.nonlocal_routes import Surgery
from medium_coupled.meet import SideGrid
from medium_frontier.nodal import discover, orbit
from pareto_easy import BudgetExpired, arrival


class Topology(Surgery):
    def normal_spines(self, spine):
        """Repair a geometrically exceptional adjacency before adding a fifth crossing."""
        bad = [i for i,(a,b) in enumerate(pairwise(spine)) if not self.normal_edges[a,b]]
        if not bad:
            return [spine]
        if len(bad) != 1:
            return []
        i = bad[0]
        routes = set()
        for radius in range(3, 8):
            left, right = max(0,i-radius), min(len(spine)-1,i+1+radius)
            nodes = spine[left:right+1]
            rank = {node:i for i,node in enumerate(nodes)}
            end = nodes[-1]

            def visit(path, unused):
                if len(routes) >= 10:
                    return
                if not unused:
                    if self.normal_edges[path[-1],end]:
                        routes.add((*spine[:left],*path,end,*spine[right+1:]))
                    return
                a = path[-1]
                choices = sorted((b for b in unused if self.normal_edges[a,b]),
                                 key=lambda b: abs(rank[b]-rank[a]-1))
                for b in choices:
                    visit((*path,b), unused-{b})

            visit(nodes[:1],set(nodes[1:-1]))
            if len(routes) >= 10:
                break
        return sorted(routes)

    def layouts(self):
        cut = [-1, *self.shells, len(self.path) - 1]
        pieces = [tuple(self.path[a+1:b+1]) for a, b in pairwise(cut)]
        low, high = pieces[::2], pieces[1::2]
        options = defaultdict(list)
        expected = self.udp.n_dv_exception
        spines = list(product(*[self.normal_spines(spine) for spine in high])) if self.args.normalize_passage else [high]
        self.state["normalized_high_spines"] = len(spines) if self.args.normalize_passage else None
        for groups, passages, directions, spines_pair in product(permutations(range(3)),
                                                     permutations(range(2)),
                                                     product((False, True), repeat=2), spines):
            a, b, c = groups
            first, second = [spines_pair[i][::-1] if rev else spines_pair[i]
                             for i, rev in zip(passages, directions)]
            shapes = [("low-low", (), first, second, ())]
            shapes += [("high-low", first[:j], first[j:], second, ())
                       for j in (1, 2, 4, 8, 12) if j < len(first)]
            shapes += [("low-high", (), first, second[:-j], second[-j:])
                       for j in (1, 2, 4, 8, 12) if j < len(second)]
            for mode, head, left, right, tail in shapes:
                if self.args.normalize_passage and mode == "low-low":
                    continue
                high_floor = sum(self.route_floor(segment) for segment in (head, left, right, tail))
                mandatory = 4 + bool(head or tail)
                if high_floor + mandatory > expected:
                    continue
                entry_exit = ((head[-1] if head else None, left[0]),
                              (left[-1], right[0]),
                              (right[-1], tail[0] if tail else None))
                choices = [self.low_rotations(low[index], *ends)
                           for index, ends in zip(groups, entry_exit)]
                if any(not routes for routes in choices):
                    continue
                key = mode, groups
                for rank in range(min(3, *[len(routes) for routes in choices])):
                    price = sum(routes[rank][0] for routes in choices)
                    starts, middle, end = [routes[rank][1] for routes in choices]
                    path = (*head, *starts, *left, *middle, *right, *end, *tail)
                    if path != tuple(self.path):
                        options[key].append((price, path))
        return {str(key): sorted(set(rows)) for key, rows in options.items()}

    def low_rotations(self, group, incoming, outgoing):
        options = []
        for route in (group, group[::-1]):
            for j in range(len(route)):
                order = (*route[j:], *route[:j])
                edges = [(incoming, order[0])] if incoming is not None else []
                if outgoing is not None:
                    edges.append((order[-1], outgoing))
                if self.route_floor(order) == 0 and all(self.allowed[a, b] for a, b in edges):
                    options.append((sum(self.geometry_floor[a, b] for a, b in edges), order))
        return sorted(options)

    def measure(self, path, step, stop, key):
        grid = SideGrid(self.udp, 0., self.args.goal, step, .013, stop, False)
        future = [0] * len(path)
        edges = list(pairwise(path))
        for i, (a, b) in reversed(list(enumerate(edges))):
            future[i] = future[i+1] + int(not self.normal_edges[a, b])
        labels = {0: (0., ())}
        latest = 0.
        try:
            for i, (a, b) in enumerate(edges):
                grid.check()
                update = {}
                for used, (release, chain) in labels.items():
                    allow = used + future[i+1] < self.udp.n_dv_exception
                    for row in grid.edges(a, b, release, 7., 9., allow):
                        if row is None:
                            continue
                        dep, flight = row[2:4]
                        dv = float(self.udp.compute_transfer(a, b, dep, flight))
                        self.count["replays"] += 1
                        spend = used + int(dv > self.normal)
                        arrive = dep + flight
                        if (dep < release - 1e-9 or dv > self.cap or arrive > self.args.goal
                                or spend + future[i+1] > self.udp.n_dv_exception):
                            continue
                        if spend not in update or arrive < update[spend][0]:
                            update[spend] = arrive, (*chain, (a,b,dep,flight,dv))
                if not update and getattr(self.args,"wide_retime",False):
                    for used,(release,chain) in labels.items():
                        slack = self.udp.n_dv_exception-used-future[i+1]
                        found,_,calls = discover(self.udp,self.orbits,a,b,release,
                            min(self.args.goal,release+24.),self.cap if slack else self.normal,
                            min(stop,time.monotonic()+2.))
                        self.count["nodal_fallback_evaluations"] += calls
                        for leg in found:
                            e = used+int(leg[-1]>self.normal)
                            finish = arrival(leg)
                            if e+future[i+1]<=self.udp.n_dv_exception and (
                                    e not in update or finish<update[e][0]):
                                update[e] = finish,(*chain,leg)
                if not update:
                    return {"transfers": i, "edge": [a,b], "last_arrival": latest}
                labels = update
                latest = min(value[0] for value in labels.values())
            for _, chain in labels.values():
                self.submit(self.build(chain), "alternate_shell_topology", key)
            return {"transfers": len(edges), "earliest": latest}
        except BudgetExpired:
            return {"transfers": i, "last_arrival": latest, "time_cap": True}
        finally:
            self.count.update(grid.stats)

    def run(self):
        if self.args.wide_retime:
            self.orbits = [orbit(self.udp,node) for node in range(self.udp.n_tomatoes)]
        options = self.layouts()
        self.state["geometry_candidates"] = {key: len(rows) for key, rows in options.items()}
        self.save()
        applied = {tuple(row["key"]) for row in self.state["trials"]}
        for rank in range(self.args.layouts):
            for family, rows in options.items():
                if rank >= len(rows) or time.monotonic() >= self.stop:
                    continue
                key = family, rank
                if key in applied:
                    continue
                _, path = rows[rank]
                start = time.monotonic()
                out = self.measure(path, .053 if rank % 2 == 0 else .035,
                                   min(self.stop, start + self.args.trial_seconds), key)
                self.state["trials"].append({"key": key, "seconds": time.monotonic()-start, **out})
                self.save()
                print(key, out, flush=True)
        self.state["wall_seconds"] = time.monotonic() - self.started
        self.save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--seed-record", type=Path, required=True)
    p.add_argument("--goal", type=float, default=182.25426732810323)
    p.add_argument("--seconds", type=float, default=85.)
    p.add_argument("--trial-seconds", type=float, default=6.)
    p.add_argument("--layouts", type=int, default=3)
    p.add_argument("--normalize-passage", action="store_true")
    p.add_argument("--wide-retime", action="store_true")
    p.set_defaults(mode="measured_spine_topology", beam=None, phase=None, anchors=(),
                   baseline_record=None)
    Topology(p.parse_args()).run()
