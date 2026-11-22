"""Seed fresh Lambert basins at mutual orbital nodes, then restore full tours."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from continuous_windows_easy import FILES, SEED, WindowSearch
from elite_easy import scan_archive
from global_dp_easy import Grid, evaluator
from global_event_easy import Topology, geometry
from kttsp_esa_solver import validate
from pareto_easy import BudgetExpired, write_json


def mean_anomaly(true, eccentricity):
    eccentric = math.atan2(math.sqrt(1-eccentricity**2)*math.sin(true),
                           eccentricity+math.cos(true))
    return eccentric-eccentricity*math.sin(eccentric)


class Search:
    cost = WindowSearch.cost

    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + args.seconds
        self.module = evaluator()
        self.udp = self.module.udp
        self.stats = Counter()
        self.rng = random.Random(args.seed)
        allowed, floor = geometry(self.udp, self.module.MU_MOON)
        self.archive, self.stats["validated_archive_files"] = scan_archive(self.udp)
        self.topology = Topology(self.udp, allowed, floor, self.archive)
        raw = json.loads(args.incumbent.read_text())
        self.reference = tuple(raw["path"])
        self.baseline = validate(self.udp, self.reference, raw["transfers"])
        if not self.baseline or not self.baseline["valid"]:
            raise ValueError("ESA rejected incumbent")
        self.best = self.baseline["fitness"][0]
        self.best_feasible = None
        self.feasible_routes = set()
        self.seen = set()
        self.attempts = []
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
        axis, self.ecc, inc, raan, peri, _ = self.udp.tomato_orbital_parameters.T
        self.period = 2*np.pi*np.sqrt(axis**3/self.module.MU_MOON)/86400.
        p = np.column_stack((np.cos(raan), np.sin(raan), np.zeros(len(axis))))
        q = np.column_stack((-np.sin(raan)*np.cos(inc), np.cos(raan)*np.cos(inc), np.sin(inc)))
        self.p = p*np.cos(peri[:, None])+q*np.sin(peri[:, None])
        self.q = q*np.cos(peri[:, None])-p*np.sin(peri[:, None])
        self.h = np.cross(p, q)
        self.grids = {}
        self.initial = [self.anomaly(i, orbit.eph(0.)[0]) for i, orbit in enumerate(self.udp.tomatoes)]
        self.group = [0]*len(axis)
        low_groups = sorted((sorted(g) for g in self.topology.cuts if len(g)==3))
        for k, group in enumerate(low_groups, 1):
            for node in group:
                self.group[node] = k

    def anomaly(self, index, vector):
        return mean_anomaly(math.atan2(self.q[index]@vector, self.p[index]@vector), self.ecc[index])

    @lru_cache(maxsize=3000)
    def phases(self, a, b):
        line = np.cross(self.h[a], self.h[b])
        if np.linalg.norm(line) < 1e-9:
            return (), ()
        return tuple(tuple(((self.anomaly(i, sign*line)-self.initial[i])/(2*np.pi)) % 1.
                           for sign in (1, -1)) for i in (a,b))

    def epochs(self, node, phases, left, right):
        period = self.period[node]
        for phase in phases:
            first = math.ceil(left/period-phase)
            last = math.floor(right/period-phase)
            for rev in range(first, last+1):
                yield (rev+phase)*period

    def classify(self, path):
        return tuple(self.group[node] for i, node in enumerate(path)
                     if i==0 or self.group[node] != self.group[path[i-1]])

    def limit(self, extra):
        return self.udp.dv_threshold_exception if extra else self.udp.dv_threshold

    def branch_options(self, a, b, clock, backwards, extra, adaptive):
        """Sample absolute nodal epochs, including both antipodes and many periods."""
        i, j = (b,a) if backwards else (a,b)
        phases = self.phases(i,j)
        scale = min(self.period[i], 2.)
        horizon = (1.65 if adaptive else .95)
        sense = -1 if backwards else 1
        stop = clock+sense*horizon
        lo, hi = sorted((clock, stop))
        centers = list(self.epochs(i, phases[0], lo-.25*scale, hi+.25*scale))
        departures = {clock+sense*x for x in (0., .17, .4, .75)}
        departures.update(center+off*scale for center in centers
                          for off in (-.16, 0., .16))
        choices = []
        min_time = self.udp.min_tof
        for d in sorted(departures):
            if d < 0 or not lo-1e-8 <= d <= hi+1e-8:
                continue
            end_interval = sorted((d+sense*min_time, d+sense*7.6))
            times = [abs(end-d) for end in self.epochs(j, phases[1], *end_interval)]
            times += [.16, .37, .7, 1.15, 1.85, 2.7, 3.6, 4.65, 5.75, 7.1]
            for f in times:
                dep = d-f if backwards else d
                end = dep+f
                if dep < 0. or f < min_time or end > self.udp.max_time:
                    continue
                dv = self.cost(a,b,dep,f)
                if math.isfinite(dv):
                    excess = max(0., dv-self.limit(extra))
                    choices.append((d+sense*f, excess, (dep,f,dv)))
        if adaptive:
            limit = self.limit(extra)
            seeded = sorted(choices, key=lambda row: sense*row[0]+.095*row[1])[:5]
            for _, _, (dep,f,_) in seeded:
                d = dep+f if backwards else dep
                bounds = [(max(0.,lo,d-.22),min(hi,d+.22)),
                          (max(min_time,f-.42),min(8.,f+.42))]

                def evaluate(x):
                    epoch, flight = map(float,x)
                    start = epoch-flight if backwards else epoch
                    if start < 0. or start+flight > self.udp.max_time:
                        return None
                    dv = self.cost(a,b,start,flight)
                    return epoch+sense*flight, max(0.,dv-limit), (start,flight,dv)

                def score(x):
                    row = evaluate(x)
                    return 1e4 if row is None else sense*(row[0]-clock)+.13*row[1]

                initial = np.clip((d,f),[b[0] for b in bounds],[b[1] for b in bounds])
                fit = minimize(score,initial,bounds=bounds,method="Powell",
                               options={"maxfev":48,"xtol":.0002})
                row = evaluate(fit.x)
                if row is not None and math.isfinite(row[2][-1]):
                    choices.append(row)
            # Independently test fresh exact branches on a denser grid in
            # the same clock direction as the nodal search.
            key = (backwards, self.latest if backwards else 0.)
            if key not in self.grids:
                self.grids[key] = Grid(self.module,.0625,.021,self.deadline,
                                       finish_day=self.latest if backwards else None,
                                       directional=True)
                self.grids[key].horizon = self.latest if backwards else min(self.udp.max_time,138.)
            grid = self.grids[key]
            release = self.latest-clock if backwards else clock
            grid_choices = grid.edges(i,j,release,2.5,7.75,bool(extra))
            if self.args.feasibility_first and not any(l is not None for l in grid_choices):
                grid_choices += grid.edges(i,j,release,4.75,10.,bool(extra))
            for leg in grid_choices:
                if leg is None:
                    continue
                _,_,start,flight,_ = leg
                dep = self.latest-start-flight if backwards else start
                if dep < 0.:
                    continue
                dv = self.cost(a,b,dep,flight)
                if math.isfinite(dv):
                    choices.append((dep if backwards else dep+flight,max(0.,dv-limit),(dep,flight,dv)))
        return choices

    def construct(self, path, extras, clock, backwards, adaptive):
        self.latest = clock
        pairs = list(zip(path,path[1:]))
        if backwards:
            pairs.reverse()
        states = [(clock, 0., ())]
        sense = -1 if backwards else 1
        for a,b in pairs:
            results = []
            for day, violation, legs in states:
                options = self.branch_options(a,b,day,backwards,(a,b) in extras,adaptive)
                for end, excess, values in options:
                    penalty = violation+excess+.018*excess**2
                    results.append((end, penalty, legs+((a,b,*values),)))
            if not results:
                return None
            next_states = []
            cells = set()
            weight = (8. if self.args.feasibility_first else .07) if adaptive else .035
            for row in sorted(results, key=lambda s: sense*s[0]+weight*s[1]):
                cell = round(row[0]*5)
                if cell in cells:
                    continue
                cells.add(cell)
                next_states.append(row)
                if len(next_states) == (3 if adaptive else 2):
                    break
            states = next_states
        return states[0][2][::-1] if backwards else states[0][2]

    def trajectory(self, path, extra, seed, adaptive, trial):
        """Restore all ΔV and chronology constraints on a full, possibly infeasible tour."""
        pairs = list(zip(path,path[1:]))
        n = len(pairs)
        x0 = np.asarray([leg[2:4] for leg in seed]).ravel()
        limits = np.array([self.limit(pair in extra)-3e-6 for pair in pairs])
        chrono = np.zeros((n-1,2*n+n))
        for i in range(n-1):
            chrono[i,2*i:2*i+3] = (-1.,-1.,1.)
        width = np.tile([2.5,1.8] if adaptive else [1.4,1.05],n)
        bounds = Bounds(np.r_[np.maximum(x0-width,np.tile([0.,self.udp.min_tof],n)),np.zeros(n)],
                        np.r_[np.minimum(x0+width,self.udp.max_time),np.full(n,np.inf)])
        cache_x = None
        cache_values = None

        def costs(x):
            nonlocal cache_x, cache_values
            if cache_x is None or not np.array_equal(cache_x,x[:2*n]):
                cache_x = x[:2*n].copy()
                cache_values = np.array([self.cost(a,b,d,f) for (a,b),(d,f)
                                         in zip(pairs,cache_x.reshape(-1,2))])
            return cache_values

        def residual(x):
            return (limits-costs(x))/100.+x[2*n:]

        def jac(x):
            values = costs(x)
            matrix = np.zeros((n,3*n))
            h = 2e-7
            for i, (a,b) in enumerate(pairs):
                d,f = x[2*i:2*i+2]
                matrix[i,2*i:2*i+2] = [(values[i]-self.cost(a,b,d+h,f))/(100*h),
                                       (values[i]-self.cost(a,b,d,f+h))/(100*h)]
            matrix[:,2*n:] = np.eye(n)
            return matrix

        def pack(x):
            return tuple((a,b,float(d),float(f),float(dv)) for (a,b),(d,f),dv
                         in zip(pairs,x[:2*n].reshape(-1,2),costs(x)))

        best_iterate = None
        best_merit = (math.inf, math.inf)

        def record(x):
            nonlocal best_iterate,best_merit
            chronology_error = np.maximum(0.,-(chrono@x)-1e-6).sum()
            violation = float(np.maximum(0.,costs(x)-limits).sum())
            merit = (violation+1000.*chronology_error, float(sum(x[2*n-2:2*n])))
            if merit < best_merit:
                best_merit = merit
                best_iterate = x.copy()
            if np.max(x[2*n:]) > 8e-8 or np.min(chrono@x) < -1e-6:
                return
            if np.any(costs(x)>limits+2e-6):
                return
            self.submit(path,pack(x),trial)

        init = np.r_[x0,np.zeros(n)]
        init[2*n:] = np.maximum(0.,-residual(init))
        trial["seed_violation_m_s"] = float(sum(init[2*n:])*100)
        record(init)
        stages = ((6., .25), (45., 1.)) if adaptive else ((23., .7),)
        for penalty, finish_weight in stages:
            weights = np.r_[np.tile([0.,.013],n),np.full(n,penalty)]
            weights[2*n-2:2*n] += finish_weight
            result = minimize(lambda x: float(weights@x), init, jac=lambda x: weights,
                              bounds=bounds, method="SLSQP", callback=record,
                              constraints=[LinearConstraint(chrono,-7e-7,np.inf),
                                           {"type":"ineq","fun":residual,"jac":jac}],
                              options={"maxiter":30 if adaptive else 33,"ftol":2e-8})
            record(result.x)
            if adaptive:
                init = best_iterate.copy()
                init[2*n:] = np.maximum(0.,(costs(init)-limits)/100.)
            else:
                init = result.x
        trial["restored_violation_m_s"] = float(np.maximum(0., costs(init)-limits).sum())
        trial["raw_finish"] = float(sum(init[2*n-2:2*n]))
        trial["lowest_violation_iterate_m_s"] = best_merit[0]

    def submit(self, path, legs, trial):
        key = tuple(l[:4] for l in legs)
        if key in self.seen:
            return
        self.seen.add(key)
        self.stats["esa_checks"] += 1
        result = validate(self.udp,path,legs)
        if not result or not result["valid"]:
            self.stats["esa_rejected"] += 1
            return
        self.stats["esa_feasible"] += 1
        self.feasible_routes.add(path)
        days = result["fitness"][0]
        trial["verified_days"] = min(days,trial.get("verified_days",math.inf))
        entry = {"path":path,"transfers":legs,"validation":result,"experiment":dict(trial),
                 "seed":self.args.seed,"incumbent":str(self.args.incumbent)}
        if self.best_feasible is None or days < self.best_feasible["validation"]["fitness"][0]:
            self.best_feasible = entry
        if days < self.best:
            self.best = days
            self.stats["records"] += 1
            write_json(FILES/f"easy_nodal_{self.run_id}_{self.stats['records']:04}.json",entry)
            print(f"ESA record: {days:.14f}",flush=True)

    def run(self):
        n = self.udp.n_tomatoes
        reference = set(zip(self.reference,self.reference[1:]))
        archived_families = {self.classify(path) for path in self.archive}
        adaptive = self.args.adaptive
        members = sorted((min(rows,key=lambda row:row[0])[0],path,
                          min(rows,key=lambda row:row[0])[1]) for path,rows in self.archive.items())
        try:
            for i in range(self.args.attempts):
                if time.monotonic() >= self.deadline:
                    raise BudgetExpired
                if i % 4 == 0:
                    # Control: a known topology, but remove all inherited timings.
                    _,path,legs = members[(i//4)%len(members)]
                    extra = {l[:2] for l in legs if l[4]>self.udp.dv_threshold}
                else:
                    # Pin endpoints in systematically different low clusters.
                    start_groups = (1,2,3)
                    group = start_groups[(i//2)%3]
                    end_group = start_groups[(start_groups.index(group)+1+i%2)%3]
                    start = self.rng.choice([k for k,g in enumerate(self.group) if g==group])
                    end = self.rng.choice([k for k,g in enumerate(self.group) if g==end_group])
                    if adaptive and i%4==3:
                        end = self.rng.choice([k for k,g in enumerate(self.group) if g==0])
                    fit = self.topology.propose(self.rng,min(self.deadline,time.monotonic()+9),
                                                reference, 13 if adaptive else 18, 2.6,
                                                forced=((n,start),(end,n)))
                    if fit is None:
                        self.stats["topology_misses"] += 1
                        continue
                    path, exceptions, _ = fit
                    extra = set(exceptions)
                backwards = adaptive and i%3!=0
                if self.args.feasibility_first:
                    clock = (121., 119., 128.)[i%3] if backwards else (0., .35, 1.1)[i%3]
                else:
                    clock = (117., 117., 123.)[i%3] if backwards else (0., .35, 1.1)[i%3]
                trial = {"attempt":i,"path":path,"family":self.classify(path),
                         "new_family":self.classify(path) not in archived_families,
                         "changed_arcs":len(set(zip(path,path[1:]))-reference),
                         "exceptions":sorted(extra),"backwards":backwards,"clock":clock}
                self.attempts.append(trial)
                legs = self.construct(path,extra,clock,backwards,adaptive)
                if legs:
                    self.stats["complete_nodal_seeds"] += 1
                    self.trajectory(path,extra,legs,adaptive,trial)
                print(json.dumps({k:v for k,v in trial.items() if k!="path"}),flush=True)
        except BudgetExpired:
            self.stats["deadline"] += 1
        finally:
            report = {"baseline":self.baseline,"best_days":self.best,
                      "best_generated":self.best_feasible,"attempts":self.attempts,
                      "counts":dict(self.stats),"distinct_feasible_orders":len(self.feasible_routes),
                      "elapsed_seconds":time.monotonic()-self.started,
                      "configuration":{**vars(self.args),"incumbent":str(self.args.incumbent)}}
            file = FILES/f"easy_nodal_{self.run_id}_summary.json"
            write_json(file,report)
            print(f"{dict(self.stats)} | {file}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent",type=Path,default=SEED)
    parser.add_argument("--seed",type=int,default=5110)
    parser.add_argument("--seconds",type=float,default=80.)
    parser.add_argument("--attempts",type=int,default=8)
    parser.add_argument("--adaptive",action="store_true")
    parser.add_argument("--feasibility-first",action="store_true")
    Search(parser.parse_args()).run()
