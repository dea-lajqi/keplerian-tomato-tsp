"""Checkpointed Small crossover search with deadline-constrained crossing blocks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, differential_evolution, least_squares, minimize
from scipy.sparse import csr_matrix, eye, lil_matrix, vstack

from continuous_windows_easy import FILES, SEED, WindowSearch
from elite_easy import anchor_bank, scan_archive
from focus_easy import retime
from global_dp_easy import Grid, evaluator
from global_event_easy import Topology, geometry
from kttsp_esa_solver import validate
from pareto_easy import BudgetExpired, arrival, write_json


class Search:
    cost = WindowSearch.cost

    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        self.started = time.monotonic()
        self.deadline = self.started + args.seconds
        self.module = evaluator()
        self.udp = self.module.udp
        self.stats = Counter()
        self.allowed, floor = geometry(self.udp, self.module.MU_MOON)
        self.archive, _ = scan_archive(self.udp)
        self.topology = Topology(self.udp, self.allowed, floor, self.archive)
        self.anchors = anchor_bank(self.archive)
        raw = json.loads(args.incumbent.read_text())
        self.reference = tuple(raw["path"])
        self.baseline = validate(self.udp, self.reference, raw["transfers"])
        if not self.baseline or not self.baseline["valid"]:
            raise ValueError("ESA rejected incumbent")
        self.best = self.baseline["fitness"][0]
        self.best_legs = tuple(tuple(l) for l in raw["transfers"])
        self.population = {path: min(rows, key=lambda row: row[0])[:2]
                           for path, rows in self.archive.items()}
        self.population[self.reference] = (self.best,self.best_legs)
        self.shell = self.udp.tomato_orbital_parameters[:,0] > 1e7
        self.history = []
        self.seen = set()
        self.orders = set()
        self.best_generated = None
        self.best_novel = None
        self.grids = {}
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
        self.iteration = 0
        if args.resume:
            old = json.loads(args.resume.read_text())
            self.rng.setstate((old["rng"][0],tuple(old["rng"][1]),old["rng"][2]))
            self.iteration = old["next_iteration"]
            for path, schedule in old["population"]:
                key = tuple(path)
                verified = validate(self.udp,key,schedule)
                if verified and verified["valid"]:
                    value = verified["fitness"][0]
                    legs = tuple(map(tuple,schedule))
                    self.population[key] = (value,legs)
                    if value < self.best:
                        self.best,self.best_legs = value,legs
            self.topology.cuts.update(frozenset(group) for group in old.get("topology_cuts", ()))
            self.topology.forbidden.extend(set(map(tuple, links)) for links in old["excluded_topologies"])

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired

    def grid(self,step=.0625,phase=0.):
        key = step,phase
        if key not in self.grids:
            self.grids[key] = Grid(self.module,step,phase,self.deadline,directional=True)
            self.grids[key].horizon = min(self.udp.max_time, self.best+18.)
        return self.grids[key]

    def checkpoint(self):
        data = {"next_iteration":self.iteration,"rng":self.rng.getstate(),
                "baseline":self.baseline,"best_days":self.best,
                "best_generated":self.best_generated,"best_novel":self.best_novel,
                "history":self.history,"counts":dict(self.stats),"distinct_orders":len(self.orders),
                "population":[(p,v[1]) for p,v in self.population.items()],
                "excluded_topologies":[sorted(x) for x in self.topology.forbidden],
                "topology_cuts":sorted(sorted(int(i) for i in group) for group in self.topology.cuts),
                "elapsed_seconds":time.monotonic()-self.started,
                "configuration":{k:str(v) if isinstance(v,Path) else v for k,v in vars(self.args).items()}}
        write_json(FILES/f"easy_paired_{self.run_id}_checkpoint.json",data)

    def submit(self, legs, trial, stage):
        key = tuple(l[:4] for l in legs)
        if key in self.seen:
            return
        self.seen.add(key)
        path = (legs[0][0],*(l[1] for l in legs))
        checked = validate(self.udp,path,legs)
        self.stats["esa_checks"] += 1
        if not checked or not checked["valid"]:
            self.stats["esa_rejected"] += 1
            return
        self.stats["esa_feasible"] += 1
        self.orders.add(path)
        days = checked["fitness"][0]
        trial["valid_days"] = min(days,trial.get("valid_days",math.inf))
        info = {"path":path,"transfers":legs,"validation":checked,"stage":stage,
                "experiment":dict(trial),"seed":self.args.seed,"incumbent":str(self.args.incumbent)}
        if self.best_generated is None or days < self.best_generated["validation"]["fitness"][0]:
            self.best_generated = info
        if path not in self.archive and (self.best_novel is None or
                                         days < self.best_novel["validation"]["fitness"][0]):
            self.best_novel = info
        old = self.population.get(path)
        if old is None or days < old[0]:
            self.population[path] = (days,tuple(legs))
        if len(self.population)>48:
            del self.population[max(self.population,key=lambda p:self.population[p][0])]
        if days < self.best:
            self.best, self.best_legs = days,tuple(legs)
            self.stats["records"] += 1
            file = FILES/f"easy_paired_{self.run_id}_{self.stats['records']:04d}.json"
            write_json(file,info)
            print(f"ESA record {days:.14f} | {file.name}",flush=True)

    def limits(self, pairs, original, available):
        """Consider moving the spare exception to the legs closest to a useful gain."""
        mandatory = {i for i,(a,b) in enumerate(pairs) if not self.allowed[0][a,b]}
        current = {i for i,l in enumerate(original) if l[4]>self.udp.dv_threshold}
        variants = [mandatory | current, mandatory]
        if len(mandatory)<available:
            variants.extend(mandatory | {i} for i in sorted(set(range(len(pairs)))-mandatory,
                            key=lambda i:-original[i][3])[:3])
        seen = set()
        for v in variants:
            key = tuple(sorted(v))
            if key not in seen and len(v)<=available:
                seen.add(key)
                yield np.array([self.udp.dv_threshold_exception if i in v else self.udp.dv_threshold
                                for i in range(len(pairs))])-3e-7

    def fit(self, pairs, initial, release, due, limits, radius, trial, deadline_attempt=True):
        """Sparse feasibility restoration followed by constrained earliest arrival."""
        n = len(pairs)
        x0 = np.asarray(initial).ravel()
        lo = np.maximum(x0-np.tile([radius,radius*.7],n),np.tile([0.,self.udp.min_tof],n))
        hi = np.minimum(x0+np.tile([radius,radius*.7],n),self.udp.max_time)
        lo[0] = min(hi[0]-1e-10,max(lo[0],release-7e-7))
        matrix = np.zeros((n,2*n))
        for i in range(n-1):
            matrix[i,2*i:2*i+3] = [-1.,-1.,1.]
        matrix[-1,-2:] = [-1.,-1.]
        rhs = np.r_[np.full(n-1,-7e-7),-due]
        dv_pattern = lil_matrix((n,2*n),dtype=int)
        for i in range(n):
            dv_pattern[i,2*i:2*i+2] = 1
        pattern = vstack((dv_pattern, csr_matrix(matrix != 0), eye(2*n)), format="csr")
        last_x, last_dv = None, None

        def dv(x):
            nonlocal last_x,last_dv
            if last_x is None or not np.array_equal(last_x,x):
                last_x=x.copy()
                last_dv=np.array([self.cost(a,b,d,f) for (a,b),(d,f) in zip(pairs,x.reshape(-1,2))])
            return last_dv

        def residual(x):
            return np.r_[np.maximum(0., (dv(x)-limits)/12.),
                         np.maximum(0.,rhs-matrix@x)*8., .012*(x-x0)]

        self.stats["block_solves"] += 1
        init = np.clip(x0,lo,hi)
        if deadline_attempt:
            fit = least_squares(residual,init,bounds=(lo,hi),jac_sparsity=pattern,
                                max_nfev=28,ftol=1e-5,gtol=1e-5)
            init = fit.x
        caps = rhs.copy()
        # Earliest arrival restoration can still recover useful profiles when
        # the original continuation deadline is unattainable.
        caps[-1] = -min(self.udp.max_time,due+radius)
        weights = np.zeros(2*n)
        weights[-2:] = 1
        def jac(x):
            base = dv(x)
            result = np.zeros((n,2*n))
            h = 2e-7
            for i,(a,b) in enumerate(pairs):
                d,f = x[2*i:2*i+2]
                result[i,2*i:2*i+2] = [(base[i]-self.cost(a,b,d+h,f))/(100*h),
                                        (base[i]-self.cost(a,b,d,f+h))/(100*h)]
            return result
        def pack(x):
            costs = dv(x)
            if (x[0]<release-9e-7 or min((matrix@x-caps)[:-1],default=0.) < -2.5e-7
                    or np.any(costs>limits+1e-7)):
                return None
            return tuple((a,b,float(d),float(f),float(c)) for (a,b),(d,f),c
                         in zip(pairs,x.reshape(-1,2),costs))
        result = minimize(lambda x:float(weights@x),init,jac=lambda _:weights,method="SLSQP",
                          bounds=Bounds(lo,hi),
                          constraints=[LinearConstraint(matrix,caps,np.inf),
                                       {"type":"ineq","fun":lambda x:(limits-dv(x))/100.,"jac":jac}],
                          options={"maxiter":26,"ftol":2e-9})
        options = [leg for x in (init,result.x) if (leg:=pack(x)) is not None]
        if not options:
            return None
        self.stats["feasible_blocks"] += 1
        return min(options,key=lambda legs:arrival(legs[-1]))

    def evolve_block(self, pairs, original, release, due, limits, radius):
        """Explore disconnected timing basins in chronological wait/flight coordinates."""
        x0 = []
        previous = release
        bounds = []
        for l in original:
            wait = max(0., l[2]-previous)
            x0 += [wait,l[3]]
            bounds += [(0.,wait+radius*.7),
                       (max(self.udp.min_tof,l[3]-radius), min(10.,l[3]+radius))]
            previous = arrival(l)
        n = len(pairs)
        initial = np.asarray(x0)
        lower,upper = np.array(bounds).T
        # Mix close and broad candidates. Keeping a feasible parent in the
        # population allows selection to reject destructive mutations.
        rng = np.random.default_rng(self.rng.randrange(2**32))
        population = rng.uniform(lower,upper,size=(max(28,5*n),2*n))
        population[:len(population)//2] = np.clip(
            initial + rng.normal(0.,.23,(len(population)//2,2*n)),lower,upper)
        population[0] = initial

        def decode(x):
            steps = x.reshape(-1,2)
            prefix = release + np.r_[0.,np.cumsum(steps.sum(axis=1))[:-1]]
            return np.column_stack((prefix+steps[:,0],steps[:,1]))

        def score(x):
            over = np.array([max(0.,self.cost(a,b,d,f)-limit)
                             for (a,b),(d,f),limit in zip(pairs,decode(x),limits)])
            tardiness = max(0.,release+sum(x)-due)
            return float(.26*sum(x)+.7*tardiness+.065*over.sum()+.002*(over@over))

        self.stats["evolution_runs"] += 1
        result = differential_evolution(score,bounds,init=population,seed=rng,
                                        mutation=(.4,1.25),recombination=.78,
                                        maxiter=48,polish=False,tol=.0002)
        reference_score = score(initial)
        self.stats["evolution_evaluations"] += result.nfev
        if result.fun < reference_score-1e-6:
            self.stats["evolution_merit_gains"] += 1
        return decode(result.x)

    def finish(self,path,seed,prefix,trial):
        anchors = dict(self.anchors)
        for l in seed:
            anchors[l[:2]] = (*anchors.get(l[:2],()),l)
        grid = self.grid(.0625,(self.iteration%3)*.014)
        schedules,depth = retime(grid,path,anchors,prefix=prefix,wait=4.75,flight=9.)
        trial["suffix_depth"] = max(depth,trial.get("suffix_depth",0))
        for legs in schedules:
            self.submit(legs,trial,"crossing block and propagated suffix")

    def crossing_trial(self):
        sorted_pool = sorted(self.population.items(),key=lambda row:row[1][0])
        band = min(len(sorted_pool),4 if self.iteration%3 else 16)
        path,(score,parent) = sorted_pool[min(band-1,int(self.rng.random()**1.6*band))]
        cross = [i for i,l in enumerate(parent) if self.shell[l[0]]!=self.shell[l[1]]]
        center = cross[self.iteration%len(cross)]
        partner = min(cross,key=lambda i:abs(i-center) if i!=center else 1000)
        paired = abs(partner-center)<=5 and self.iteration%2==0
        size = (2,3,4)[self.iteration%3]
        left = max(0,min(center,partner) - size if paired else center-size)
        right = min(len(parent),max(center,partner)+size+1 if paired else center+size+1)
        local = parent[left:right]
        release = arrival(parent[left-1]) if left else 0.
        pairs = tuple(l[:2] for l in local)
        outside = sum(l[4]>self.udp.dv_threshold for l in (*parent[:left],*parent[right:]))
        old_finish = arrival(local[-1])
        gap = (.15,.45,.85,1.4,2.1)[(self.iteration//3)%5]
        due = max(release + len(local)*self.udp.min_tof,old_finish-gap)
        radius = (1.0,1.7,2.6)[self.iteration%3]
        trial = {"iteration":self.iteration,"kind":"crossing","crossings":[list(parent[k][:2]) for k in cross if left<=k<right],
                 "left":left,"right":right,"old_block_end":old_finish,"block_deadline":due,"parent_days":score}
        self.history.append(trial)
        for variant,limits in enumerate(self.limits(pairs,local,self.udp.n_dv_exception-outside)):
            if variant>=self.args.variants:
                break
            for restart in range(2 if variant==0 else 1):
                initial = np.array([l[2:4] for l in local])
                if restart:
                    initial[:,0] -= np.linspace(0.,gap,len(local))
                    initial[:,1] -= np.maximum(0.,initial[:,1]-.12)*.09
                if self.args.evolve and restart:
                    initial = self.evolve_block(pairs,local,release,due,limits,radius)
                block = self.fit(pairs,initial,release,due,limits,radius,trial)
                if block is None:
                    continue
                end = arrival(block[-1])
                trial["best_block_end"] = min(end,trial.get("best_block_end",math.inf))
                prefix = (*parent[:left],*block)
                if outside + sum(l[4]>self.udp.dv_threshold for l in block)<=self.udp.n_dv_exception:
                    candidate = (*prefix,*parent[right:])
                    if right==len(parent) or end<=parent[right][2]+1e-6:
                        self.submit(candidate,trial,"paired crossing")
                    if end < old_finish-.015 or variant>0:
                        self.finish(path,parent,prefix,trial)

    def topology_trial(self):
        reference = set(zip(self.reference,self.reference[1:]))
        count = self.udp.n_tomatoes
        low = [tuple(sorted(g)) for g in self.topology.cuts if len(g)==3]
        self.rng.shuffle(low)
        forced = () if self.iteration%12==0 else ((count,self.rng.choice(low[0])),
                                                    (self.rng.choice(low[1]),count))
        fit = self.topology.propose(self.rng,min(self.deadline,time.monotonic()+12.),reference,
                                     (7,12,18)[(self.iteration//4)%3],2.2,forced=forced)
        trial = {"iteration":self.iteration,"kind":"topology"}
        self.history.append(trial)
        if fit is None:
            self.stats["no_static_route"] += 1
            return
        path,exception,_ = fit
        trial["path"] = path
        trial["exceptions"] = exception
        trial["changed_arcs"] = len(set(zip(path,path[1:]))-reference)
        for step,phase in ((.0625,.012),(.046875,.021)):
            grid = self.grid(step,phase)
            schedules,depth = retime(grid,path,self.anchors,wait=6.,flight=10.)
            trial["visits_reached"] = max(depth+1,trial.get("visits_reached",0))
            for legs in schedules:
                self.submit(legs,trial,"alternative shell topology")
            if schedules:
                return

    def run(self):
        last_save = time.monotonic()
        try:
            while self.best>101. and self.iteration<self.args.iterations:
                self.check()
                if self.iteration%4==0:
                    self.topology_trial()
                else:
                    self.crossing_trial()
                self.iteration += 1
                if time.monotonic()-last_save>=30:
                    self.checkpoint()
                    last_save = time.monotonic()
                    print(f"{self.iteration} | {self.best:.12f} days | {dict(self.stats)}",flush=True)
        except BudgetExpired:
            pass
        finally:
            self.checkpoint()
            print(f"{self.iteration} | {self.best:.12f} days | {dict(self.stats)} | {self.run_id}",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds",type=float,default=600.)
    p.add_argument("--iterations",type=int,default=2000)
    p.add_argument("--seed",type=int,default=9161)
    p.add_argument("--incumbent",type=Path,default=SEED)
    p.add_argument("--resume",type=Path)
    p.add_argument("--evolve",action="store_true")
    p.add_argument("--variants",type=int,default=4)
    Search(p.parse_args()).run()
