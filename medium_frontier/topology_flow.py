"""Geometry-only path cover with subtour cuts to propose five-crossing topologies."""
import argparse
from itertools import pairwise
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from medium_frontier.topology import Topology


class FlowTopology(Topology):
    def groups(self):
        bounds = [-1, *self.shells, len(self.path)-1]
        return [tuple(self.path[a+1:b+1]) for a, b in pairwise(bounds)][::2]

    def path_cover(self, groups, start_high, due):
        high = [i for i in self.path if self.axis(i) > 8e6]
        portals = [181, 182, 183]
        start, end = 184, 185
        graph = {}
        anchors = {leg[:2]: leg[3] for leg in self.legs if leg[-1] <= self.normal}
        for a in high:
            for b in high:
                if a != b and self.normal_edges[a,b]:
                    # Penalize greater plane changes. Use measured normal flights
                    # only as a search preference, never as a geometric lower bound.
                    proxy = .10 + 2.0 * (self.geometry_floor[a,b] / self.normal)**2
                    graph[a,b] = min(proxy, anchors.get((a,b), proxy))
            for portal, group in zip(portals, groups):
                low_edge = min(self.geometry_floor[a,b] for b in group)
                if any(self.allowed[a,b] for b in group):
                    price = .24 + (low_edge / self.cap)**2
                    graph[a,portal] = price
                    graph[portal,a] = price
        graph.update({(start,node):0. for node in (high if start_high else portals)})
        graph.update({(node,end):0. for node in (portals if start_high else high)})
        pairs = list(graph)
        nodes = [*high, *portals]
        out = {node: [] for node in (*nodes, start)}
        into = {node: [] for node in (*nodes, end)}
        for column, (a,b) in enumerate(pairs):
            out[a].append(column)
            into[b].append(column)
        edge_costs = np.array(list(graph.values()))
        restrictions = [(indices, 1., 1.) for indices in (*out.values(), *into.values())]
        cuts = 0
        objective = None
        for _ in range(18):
            remaining = due-time.monotonic()
            if remaining < .3:
                break
            rr, cc = [], []
            for i, (indices, _, _) in enumerate(restrictions):
                rr.extend([i]*len(indices))
                cc.extend(indices)
            matrix = coo_matrix((np.ones(len(rr)), (rr,cc)),
                                shape=(len(restrictions),len(pairs))).tocsc()
            result = milp(edge_costs, integrality=np.ones(len(pairs)),
                          bounds=Bounds(0.,1.),
                          constraints=LinearConstraint(matrix,
                              [r[1] for r in restrictions], [r[2] for r in restrictions]),
                          options={"time_limit": min(8., remaining), "mip_rel_gap": .02})
            if result.x is None:
                break
            objective = result.fun
            chosen = {a:b for (a,b),x in zip(pairs,result.x) if x > .5}
            covered, chain = {start,end}, []
            node = chosen[start]
            while node != end:
                chain.append(node)
                covered.add(node)
                node = chosen[node]
            if len(chain) == len(nodes):
                return chain, {"cuts": cuts, "proxy": objective, "status": int(result.status)}
            left = set(nodes) - covered
            while left:
                node = min(left)
                cycle = {node}
                nxt = chosen[node]
                while nxt != node:
                    cycle.add(nxt)
                    nxt = chosen[nxt]
                left -= cycle
                indices = [i for a in cycle for i in out[a] if pairs[i][1] in cycle]
                restrictions.append((indices, -np.inf, len(cycle)-1.))
                cuts += 1
            # Respect the exact required crossing count. A short low-group
            # revisit cannot arise, since each contracted portal is unique.
        return None, {"cuts": cuts, "proxy": objective}

    def run(self):
        groups = self.groups()
        tested = {trial["key"] for trial in self.state["trials"]}
        for start_high in (True, False):
            key = "high_start" if start_high else "high_end"
            if key in tested or time.monotonic() >= self.stop:
                continue
            begin = time.monotonic()
            skeleton, stats = self.path_cover(groups, start_high, min(self.stop, begin+self.args.trial_seconds))
            if skeleton is not None:
                options = []
                for i,node in enumerate(skeleton):
                    if node < 181:
                        continue
                    a = skeleton[i-1] if i else None
                    b = skeleton[i+1] if i+1 < len(skeleton) else None
                    options.append(self.low_rotations(groups[node-181], a, b))
                stats["skeleton"] = skeleton
                stats["rotation_counts"] = [len(rows) for rows in options]
                stats["timing"] = []
                for rank in range(min(3, *[len(rows) for rows in options])):
                    if time.monotonic() >= self.stop:
                        break
                    replacing = iter(rows[rank][1] for rows in options)
                    order = tuple(v for node in skeleton for v in
                                  ((node,) if node < 181 else next(replacing)))
                    stats["timing"].append(self.measure(order,.042+rank*.006,
                        min(self.stop,time.monotonic()+12.),(key,rank)))
            self.state["trials"].append({"key": key, "seconds": time.monotonic()-begin, **stats})
            self.save()
            print(key, stats, flush=True)
        self.state["wall_seconds"] = time.monotonic()-self.started
        self.save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--seed-record", type=Path, required=True)
    p.add_argument("--goal", type=float, default=182.25426732810323)
    p.add_argument("--seconds", type=float, default=145.)
    p.add_argument("--trial-seconds", type=float, default=52.)
    p.set_defaults(mode="geometry_path_flow", beam=None, phase=None, anchors=(),
                   baseline_record=None)
    FlowTopology(p.parse_args()).run()
