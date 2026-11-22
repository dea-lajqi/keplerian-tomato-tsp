"""Continuous-window mixed-integer search for the ESA Small instance.

A column is a directed transfer and a segment in departure/flight time.
The solver can pick any point on the segment. Sampled interior checks screen
segments; only the original nonlinear evaluator can accept complete tours.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import chain
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import Bounds, milp, minimize
from scipy.stats import qmc

from elite_easy import scan_archive
from global_dp_easy import evaluator
from global_event_easy import FILES, Rows, add_degrees, build_path, geometry
from kttsp_esa_solver import validate
from pareto_easy import BudgetExpired, write_json

SEED = FILES / "easy_chain_20261121T162006Z_1bb31429_0001.json"


class WindowSearch:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + args.seconds
        self.module = evaluator()
        self.udp = self.module.udp
        self.normal, self.exception = self.udp.dv_threshold, self.udp.dv_threshold_exception
        self.allowed, self.floor = geometry(self.udp, self.module.MU_MOON)
        self.rng = np.random.default_rng(args.seed)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        raw = json.loads(args.incumbent.read_text())
        self.path = tuple(raw["path"])
        self.legs = tuple(tuple(leg) for leg in raw["transfers"])
        self.reference = validate(self.udp, self.path, self.legs)
        if not self.reference or not self.reference["valid"]:
            raise ValueError("ESA rejected the incumbent")
        self.best = self.reference["fitness"][0]
        self.shell = self.udp.tomato_orbital_parameters[:, 0] > 2 * min(
            self.udp.tomato_orbital_parameters[:, 0])
        self.stats = Counter()
        self.windows = set()
        self.archived_windows = set()
        self.attempts = []
        self.geometry_cuts = []
        self.seen = set()
        self.orders = set()
        self.best_feasible = None
        self.lowest_flight = None
        self.new_edges = set()
        archive, self.stats["validated_archive_files"] = scan_archive(self.udp)
        self.archived = sorted(min(items, key=lambda row: row[0]) for items in archive.values())
        for row in self.allowed[0]:
            self.stats["normal_directed_edges"] += int(row.sum())
        unseen = set(range(self.udp.n_tomatoes))
        while unseen:
            todo = [unseen.pop()]
            group = set(todo)
            while todo:
                node = todo.pop()
                neighbors = unseen.intersection(np.flatnonzero(self.allowed[0][node]))
                todo.extend(neighbors)
                group.update(neighbors)
                unseen.difference_update(neighbors)
            self.geometry_cuts.append(group)

    def cost(self, a, b, dep, tof):
        """Same Lambert branches as ESA, with scalar norms to speed discovery."""
        if time.monotonic() >= self.deadline:
            raise BudgetExpired
        self.stats["lambert_evaluations"] += 1
        r1, v1 = self.udp.tomatoes[a].eph(float(dep))
        r2, v2 = self.udp.tomatoes[b].eph(float(dep + tof))
        best = math.inf
        pk = self.module.pk
        for cw in (False, True):
            try:
                result = pk.lambert_problem(r1, r2, float(tof) * pk.DAY2SEC,
                                            self.module.MU_MOON, cw, self.udp.max_revs)
            except Exception:
                continue
            for u, v in zip(result.get_v1(), result.get_v2()):
                best = min(best, math.dist(u, v1) + math.dist(v, v2))
        return best

    def point(self, a, b, times):
        d, f = map(float, times)
        dv = self.cost(a, b, d, f)
        if dv > self.exception + 1e-6:
            return
        self.windows.add((a, b, int(dv > self.normal), d, f, d, f))

    def support(self, a, b, times, limit, radius):
        """Trace short segments of a feasible window with continuous support solves."""
        d, f = map(float, times)
        bounds = [(max(0., d - radius), min(self.args.cap, d + radius)),
                  (max(self.udp.min_tof, f - radius), min(self.args.cap, f + radius))]
        points = []
        for objective in ((1., 1.), (-1., 0.), (0., 1.)):
            weights = np.asarray(objective)
            fit = minimize(lambda x: float(weights @ x), [d, f], jac=lambda x: weights,
                           bounds=bounds, method="SLSQP",
                           constraints=({"type": "ineq", "fun": lambda x:
                                         (limit - self.cost(a, b, *x)) / 100.},),
                           options={"maxiter": 18, "ftol": 2e-9, "eps": 2e-7})
            if self.cost(a, b, *fit.x) <= limit + 1e-7:
                points.append(fit.x)
                self.point(a, b, fit.x)
        center = np.array([d, f])
        extra = int(limit > self.normal)
        for endpoint in points:
            end = endpoint.copy()
            for _ in range(4):
                if np.linalg.norm(end - center) < 1e-5:
                    break
                if all(self.cost(a, b, *(center + t * (end - center))) <= limit + 1e-7
                       for t in (.125, .25, .5, .75, .875, 1.)):
                    self.windows.add((a, b, extra, *map(float, center), *map(float, end)))
                    break
                end = (center + end) / 2

    def seed_archive(self):
        for schedule in [self.legs, *(row[1] for row in self.archived[:12])]:
            for a, b, d, f, _ in schedule:
                if d + f <= self.args.cap + 1e-7:
                    self.point(a, b, (d, f))
        self.archived_windows = self.windows.copy()
        for a, b, d, f, dv in sorted(self.legs, key=lambda l: -l[3]):
            limit = self.normal if dv <= self.normal else self.exception
            self.support(a, b, (d, f), limit - 5e-7, .22)

    def discover(self, seconds, extreme=False, targeted=False):
        """Search independent absolute epochs and refine separated ΔV basins."""
        stop = min(self.deadline, time.monotonic() + seconds)
        pairs = list(zip(*np.nonzero(self.allowed[1])))
        archived = {w[:2] for w in self.windows}
        known_crossings = {w[:2] for w in self.windows if self.shell[w[0]] != self.shell[w[1]]}
        reference = {l[:2]: l for l in self.legs}
        def priority(pair):
            a, b = pair
            cross = self.shell[a] != self.shell[b]
            if targeted:
                return (6 * (self.allowed[0][a, b] and self.shell[a] and self.shell[b])
                        + 2 * cross + .15 * reference.get(pair, (0, 0, 0, 0))[3]
                        + self.rng.random())
            if extreme:
                return 4 * cross + 2 * self.allowed[0][a, b] + .5 * (pair not in archived) + self.rng.random()
            return 4 * (cross and pair not in known_crossings) + 2 * (pair not in archived) + self.rng.random()

        for a, b in sorted(pairs, key=priority, reverse=True):
            if time.monotonic() >= stop:
                return
            a, b = int(a), int(b)
            self.stats["discovery_edges"] += 1
            cross = self.shell[a] != self.shell[b]
            sample = qmc.Sobol(2, scramble=True, seed=int(self.rng.integers(2**32))).random_base2(
                self.args.power + int(cross and extreme))
            sample[:, 1] = self.udp.min_tof + sample[:, 1] ** 1.45 * (12 if cross else 7)
            sample[:, 0] *= self.args.cap - sample[:, 1]
            if targeted and not cross and self.allowed[0][a, b]:
                # Spend half the budget around the absolute epochs at which either
                # endpoint is used in good tours, including reversed arcs.
                epochs = [l[2] for _, route, _ in self.archived[:8] for l in route
                          if a in l[:2] or b in l[:2]]
                if epochs:
                    half = len(sample) // 2
                    sample[:half, 0] = np.clip(self.rng.choice(epochs, half)
                                                + self.rng.uniform(-8., 8., half),
                                                0., self.args.cap - sample[:half, 1])
            ranked = sorted((self.cost(a, b, *x), *x) for x in sample)
            starts = []
            for _, d, f in ranked:
                if any(abs(d - x) < 2.0 and abs(f - y) < .7 for x, y in starts):
                    continue
                starts.append((d, f))
                if len(starts) >= (12 if cross and extreme else 6):
                    break
            limits = [self.exception]
            if self.allowed[0][a, b]:
                limits.insert(0, self.normal)
            found = False
            for d, f in starts:
                bounds = [(max(0., d - 1.7), min(self.args.cap - self.udp.min_tof, d + 1.7)),
                          (max(self.udp.min_tof, f - .65), min(13., f + .65))]
                fit = minimize(lambda x: self.cost(a, b, *x), (d, f), method="Powell",
                               bounds=bounds, options={"maxfev": 95, "xtol": 1e-4})
                seed = min((self.cost(a, b, d, f), np.array([d, f])),
                           (float(fit.fun), fit.x), key=lambda item: item[0])
                if seed[0] > self.exception - 1e-5 or sum(seed[1]) > self.args.cap:
                    continue
                self.point(a, b, seed[1])
                found = True
                for limit in limits:
                    if seed[0] <= limit - 1e-5:
                        self.support(a, b, seed[1], limit - 1e-5, .45)
            if found and (a, b) not in archived:
                self.new_edges.add((a, b))

    def representative_windows(self):
        """Keep phase diversity while reducing branch-and-bound size."""
        if not self.args.sparse:
            return self.windows
        groups = defaultdict(list)
        for w in sorted(self.windows - self.archived_windows):
            a, b, exception, d, f, end_d, end_f = w
            groups[a, b, exception, int(min(d, end_d) / 10.)].append(w)
        kept = self.archived_windows & self.windows
        for group in groups.values():
            # Preserve short flight, early arrival and latest release choices.
            kept.add(min(group, key=lambda w: (min(w[4], w[6]), w[3] + w[4])))
            kept.add(min(group, key=lambda w: min(w[3] + w[4], w[5] + w[6])))
            kept.add(max(group, key=lambda w: max(w[3], w[5])))
            segments = [w for w in group if w[3:5] != w[5:7]]
            if segments:
                kept.add(max(segments, key=lambda w: abs(w[3] - w[5])))
        return kept

    def solve(self, seconds, bias, cap, new_crossings=0, different_exceptions=False,
              changed_arcs=0, relocate_high=False):
        """One binary per window and one continuous interpolation coordinate."""
        n = self.udp.n_tomatoes
        reference = set(zip(self.path, self.path[1:]))
        reference_exceptions = {l[:2] for l in self.legs if l[4] > self.normal}
        reference_crossings = {pair for pair in reference if self.shell[pair[0]] != self.shell[pair[1]]}
        # Interior points remain available through segments. Remove redundant
        # degenerate windows only under Pareto dominance in arrival and flight.
        points = defaultdict(list)
        arcs = []
        for w in sorted(self.representative_windows()):
            a, b, e, d, f, end_d, end_f = w
            if min(d + f, end_d + end_f) > cap + 1e-7:
                continue
            if (d, f) != (end_d, end_f):
                arcs.append(w)
            else:
                points[a, b, e].append(w)
        for group in points.values():
            frontier = []
            for w in sorted(group, key=lambda row: (-row[3], row[4])):
                if any(v[3] + v[4] <= w[3] + w[4] and v[4] <= w[4] for v in frontier):
                    continue
                frontier.append(w)
            arcs.extend(frontier)
        arcs.extend((n, j, 0, 0., 0., 0., 0.) for j in range(n))
        arcs.extend((j, n, 0, cap, 0., cap, 0.) for j in range(n))
        m = len(arcs)
        rows = Rows()
        out, ins = add_degrees(rows, arcs, n)
        rows.add(((k, w[2]) for k, w in enumerate(arcs)), upper=self.udp.n_dv_exception)
        if changed_arcs:
            rows.add(((k, 1) for k, w in enumerate(arcs) if w[:2] in reference),
                     upper=n - 1 - changed_arcs)
        if relocate_high:
            rows.add(((k, e) for k, (a, b, e, *_) in enumerate(arcs)
                      if a != n and b != n and self.shell[a] and self.shell[b]
                      and (a, b) not in reference_exceptions), lower=1)
        if new_crossings:
            rows.add(((k, 1) for k, (a, b, *_) in enumerate(arcs)
                      if a != n and b != n and self.shell[a] != self.shell[b]
                      and (a, b) not in reference_crossings), lower=new_crossings)
        if different_exceptions:
            rows.add(((k, e) for k, (a, b, e, *_) in enumerate(arcs)
                      if (a, b) not in reference_exceptions), lower=1)
        for group in self.geometry_cuts:
            rows.add(((k, 1) for i in group for k in out[i] if arcs[k][1] not in group), lower=1)

        def time_terms(indices, arrival=False, sign=1.):
            for k in indices:
                _, _, _, d, f, end_d, end_f = arcs[k]
                base = d + f if arrival else d
                last = end_d + end_f if arrival else end_d
                yield k, sign * base
                yield m + k, sign * (last - base)

        objective = np.zeros(2 * m + 1)
        upper = np.ones(2 * m + 1)
        upper[-1] = cap
        objective[-1] = 1.
        for k, w in enumerate(arcs):
            _, _, _, d, f, end_d, end_f = w
            objective[k] = bias * f
            objective[m + k] = bias * (end_f - f)
            if (d, f) == (end_d, end_f):
                upper[m + k] = 0.
            else:
                rows.add(((m + k, 1), (k, -1)), upper=0.)
        for i in range(n):
            # Tiny chronology allowance matches the official evaluator.
            rows.add(chain(time_terms(out[i]), time_terms(ins[i], True, -1.)), lower=-9.5e-7)
            rows.add(chain(((2 * m, 1),), time_terms(ins[i], True, -1.)), lower=0.)
            ticks = sorted({max(arcs[k][3], arcs[k][5]) for k in out[i]})
            if len(ticks) > 26:
                ticks = [ticks[j * len(ticks) // 26] for j in range(26)]
            for t in ticks:
                rows.add(chain(((k, 1) for k in out[i] if max(arcs[k][3], arcs[k][5]) <= t),
                               ((k, -1) for k in ins[i] if min(arcs[k][3] + arcs[k][4],
                                                              arcs[k][5] + arcs[k][6]) <= t + 1e-6)),
                         upper=0.)
        seconds = min(seconds, self.deadline - time.monotonic())
        if seconds <= 0:
            raise BudgetExpired
        fit = milp(objective, integrality=np.r_[np.ones(m), np.zeros(m + 1)],
                   bounds=Bounds(np.zeros(2 * m + 1), upper),
                   constraints=rows.linear(2 * m + 1),
                   options={"time_limit": seconds, "mip_rel_gap": 1e-7})
        report = {"status": int(fit.status), "cap": cap, "flight_weight": bias,
                  "required_new_crossings": new_crossings, "different_exceptions": different_exceptions,
                  "required_changed_arcs": changed_arcs, "relocate_high_exception": relocate_high,
                  "segments": int(np.count_nonzero(upper[m:2*m])),
                  "windows": m - 2*n, "rows": len(rows.lower)}
        dual_bound = getattr(fit, "mip_dual_bound", None)
        if dual_bound is not None and math.isfinite(dual_bound):
            report["window_model_dual_bound"] = float(dual_bound)
        if fit.x is not None:
            selected = np.flatnonzero(fit.x[:m] > .5)
            legs = []
            report["interior_choices"] = 0
            for k in selected:
                a, b, e, d, f, end_d, end_f = arcs[k]
                t = float(np.clip(fit.x[m+k], 0., 1.))
                report["interior_choices"] += int(1e-6 < t < 1.-1e-6)
                legs.append((a, b, d+t*(end_d-d), f+t*(end_f-f), e))
            result = build_path(legs, range(len(legs)), n)
            if result:
                path, candidate = result
                actual = tuple((*leg[:4], float(self.udp.compute_transfer(*leg[:4]))) for leg in candidate)
                invalid = [arcs[k] for k, l in zip((k for k in selected if max(arcs[k][:2]) < n),
                                                  (l for l in legs if max(l[:2]) < n))
                           if self.udp.compute_transfer(*l[:4]) >
                           (self.exception + 1e-6 if arcs[k][2] else self.normal)]
                for w in invalid:
                    self.windows.discard(w)
                    for df in (w[3:5], w[5:7]):
                        self.point(w[0], w[1], df)
                report["invalid_segments_split"] = len(invalid)
                report["finish"] = actual[-1][2] + actual[-1][3]
                report["flight"] = sum(l[3] for l in actual)
                report["changed_arcs"] = len(set(zip(path, path[1:])) - reference)
                report["exceptions"] = [l[:2] for l in actual if l[4] > self.normal]
                report["crossings"] = [l[:2] for l in actual if self.shell[l[0]] != self.shell[l[1]]]
                self.submit(path, actual, report)
        self.attempts.append(report)
        print(json.dumps(report), flush=True)

    def submit(self, path, legs, report):
        key = tuple(l[:4] for l in legs)
        if key in self.seen:
            report["duplicate"] = True
            return
        self.seen.add(key)
        self.stats["esa_checked"] += 1
        checked = validate(self.udp, path, legs)
        report["validation"] = checked
        if not checked or not checked["valid"]:
            return
        self.stats["esa_feasible"] += 1
        self.orders.add(path)
        days = checked["fitness"][0]
        if self.best_feasible is None or days < self.best_feasible:
            self.best_feasible = days
        flight = sum(l[3] for l in legs)
        if self.lowest_flight is None or flight < self.lowest_flight["flight_days"]:
            self.lowest_flight = {"flight_days": flight, "finish_days": days, "path": path,
                                  "transfers": legs, "validation": checked}
        if days < self.best:
            self.best = days
            self.stats["records"] += 1
            write_json(FILES / f"easy_windows_{self.run_id}_{self.stats['records']:04}.json",
                       {"path": path, "transfers": legs, "validation": checked,
                        "seed_file": str(self.args.incumbent), "seed": self.args.seed,
                        "experiment": report})

    def run(self):
        try:
            self.seed_archive()
            if self.args.adapted:
                self.discover(self.args.discovery, targeted=True)
                self.discover(self.args.discovery, extreme=True)
                print(f"Discovery: {len(self.windows)} windows, {len(self.new_edges)} new directed edges",
                      flush=True)
                for changes, crossings, relocate, bias, cap in (
                        (6, 2, False, .45, self.args.cap),
                        (5, 0, True, .7, self.args.cap),
                        (5, 1, False, .15, self.best - .01),
                        (0, 0, False, 1.5, self.best + .1),
                        (4, 1, False, .35, self.args.cap),
                        (0, 0, False, 0., self.best - .001)):
                    self.solve(self.args.mip, bias, cap, new_crossings=crossings,
                               changed_arcs=changes, relocate_high=relocate)
                return
            self.solve(self.args.mip, .05, self.best + .02)
            self.discover(self.args.discovery)
            print(f"Discovery: {len(self.windows)} windows, {len(self.new_edges)} new directed edges",
                  flush=True)
            self.solve(self.args.mip, .15, self.best + .05)
            self.solve(self.args.mip, 1.0, self.args.cap, new_crossings=1)
            self.discover(self.args.discovery, extreme=True)
            self.solve(self.args.mip, .5, self.args.cap, different_exceptions=True)
            self.solve(self.args.mip, .15, self.best + .0001)
        except BudgetExpired:
            self.stats["deadline"] += 1
        finally:
            summary = {"seed": self.args.seed, "incumbent": str(self.args.incumbent),
                       "best_days": self.best, "best_generated_days": self.best_feasible,
                       "elapsed_seconds": time.monotonic() - self.started,
                       "counts": dict(self.stats), "distinct_feasible_orders": len(self.orders),
                       "new_edges": sorted(self.new_edges), "attempts": self.attempts,
                       "lowest_flight_verified_schedule": self.lowest_flight,
                       "configuration": {**vars(self.args), "incumbent": str(self.args.incumbent)}}
            file = FILES / f"easy_windows_{self.run_id}_summary.json"
            write_json(file, summary)
            print(file, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent", type=Path, default=SEED)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--discovery", type=float, default=40)
    parser.add_argument("--mip", type=float, default=22)
    parser.add_argument("--power", type=int, default=8)
    parser.add_argument("--cap", type=float, default=120.)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--adapted", action="store_true")
    parser.add_argument("--seed", type=int, default=4121)
    WindowSearch(parser.parse_args()).run()
