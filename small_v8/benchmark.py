"""Controlled Small tests of measured transfer gates under explicit ΔV allocation."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_v6.search import Search
from continuous_windows_easy import FILES
from global_event_easy import Topology, geometry
from pareto_easy import BudgetExpired, arrival
from small_v2 import atomic_json


class Benchmark(Search):
    def __init__(self,args):
        super().__init__(args)
        self.checkpoint_number=0
        self.cases=[]
        self.traces=[]
        self.allocation=set(l[:2] for l in self.seed_legs if l[4]>self.limit)
        self.target=tuple(args.edge)
        self.archive=self.parents
        self.floors=geometry(self.udp,self.module.MU_MOON)[1]

    def persist(self):
        self.checkpoint_number+=1
        path=self.out/f"v8_{self.job}_{self.checkpoint_number:04d}_checkpoint.json"
        atomic_json(path,{"elapsed":time.monotonic()-self.started,"stats":dict(self.stats),
                          "best_days":self.best,"best_generated":self.best_generated,
                          "orders":sorted(self.orders),"finish_bins":sorted(self.finishes),
                          "cases":self.cases,"traces":self.traces,"seed":self.args.seed})

    def stages(self):
        routes=sorted(self.archive,key=lambda p:arrival(self.archive[p][-1]))
        if self.args.mode=="original":
            for path in routes:
                if self.target in zip(path,path[1:]) and self.allocation=={
                        l[:2] for l in self.archive[path] if l[4]>self.limit}:
                    yield path,self.allocation
            return
        # An alternative receives the fifth exception only when the other
        # four are reserved by the geometry and the shell crossing order.
        topology=Topology(self.udp,(self.normal,self.allowed),self.floors,
                          {p:[(arrival(l[-1]),l,"archive")] for p,l in self.archive.items()})
        ref=set(zip(self.seed_path,self.seed_path[1:]))
        for p in routes[:18]:
            if self.target not in zip(p,p[1:]):
                continue
            mandatory={(a,b) for a,b in zip(p,p[1:]) if not self.normal[a,b]}
            if len(mandatory|{self.target})<=5:
                yield p,mandatory|{self.target}
        for _ in range(self.args.proposals):
            self.check()
            n=self.udp.n_tomatoes
            answer=topology.propose(self.rng,min(self.deadline,time.monotonic()+4.),ref,
                4,1.9,forced=((n,self.seed_path[0]),(self.seed_path[-1],n)),
                forced_exception=self.target)
            self.stats["alternate_topology_proposals"]+=1
            if answer:
                path,exceptions,_=answer
                yield path,set(exceptions)

    def fixed(self,path,clock,allocation,reverse=False):
        legs=()
        edges=list(zip(path,path[1:]))
        if reverse:
            edges.reverse()
        for a,b in edges:
            special=(a,b) in allocation
            viable=[l for l in self.edge_options(a,b,clock,reverse,special,True)
                    if bool(l[4]>self.limit)==special]
            if not viable:
                return clock,legs,(a,b)
            edge=min(viable,key=lambda l:-l[2] if reverse else arrival(l))
            clock=edge[2] if reverse else arrival(edge)
            legs=(edge,*legs) if reverse else (*legs,edge)
        return clock,legs,None

    def discover(self,release,special,limit_seconds=2.):
        """Target the specific gate off grid, then keep the arrival frontier."""
        a,b=self.target
        limit=self.extra-1e-6 if special else self.limit-1e-6
        stop=min(self.deadline,time.monotonic()+limit_seconds)
        options=[]

        def dv(x):
            return self.cost(a,b,*x)

        def accept(x):
            d,f=map(float,x)
            value=dv(x)
            if d>=release-8e-7 and f>=self.udp.min_tof and value<=limit and (
                    (value>self.limit)==special):
                options.append((a,b,d,f,value))

        for fine in (False,True):
            for l in self.edge_options(a,b,release,False,special,fine):
                if (l[4]>self.limit)==special:
                    options.append(l)
        seed=int(self.rng.randrange(2**32))
        samples=qmc.Sobol(2,scramble=True,seed=seed).random_base2(7)
        samples[:,0]=max(0.,release-.02)+samples[:,0]*4.2
        samples[:,1]=self.udp.min_tof+samples[:,1]**1.3*8.5
        values=sorted((dv(x),tuple(x)) for x in samples)
        starts=[l[2:4] for l in options]
        starts.extend(x for _,x in values[:7])
        for d,f in starts:
            if time.monotonic()>=stop:
                break
            bounds=((max(release-7e-7,d-.42),d+.42),
                    (max(self.udp.min_tof,f-.45),min(self.udp.max_time,f+.45)))
            if bounds[0][1]<=bounds[0][0]:
                continue
            fit=minimize(dv,(max(d,bounds[0][0]),f),bounds=bounds,method="Powell",
                         options={"maxfev":65,"xtol":1e-5})
            if fit.fun>limit:
                continue
            accept(fit.x)
            constraints=[{"type":"ineq","fun":lambda x:limit-dv(x)}]
            if special:
                constraints.append({"type":"ineq","fun":lambda x:dv(x)-self.limit-1e-5})
            solved=minimize(sum,fit.x,bounds=bounds,method="SLSQP",constraints=constraints,
                            options={"maxiter":21,"ftol":2e-9,"eps":2e-7})
            accept(solved.x)
        chosen=sorted(set(options),key=arrival)
        frontier=[]
        for l in chosen:
            if any(abs(old[2]-l[2])<.08 for old in frontier):
                continue
            frontier.append(l)
            if len(frontier)>=self.args.windows:
                break
        self.stats["measured_gate_windows"]+=len(frontier)
        return frontier

    def run(self):
        try:
            for i,(route,allocation) in enumerate(self.stages()):
                self.check()
                if i>=self.args.proposals:
                    break
                location=route.index(self.target[0])
                if route[location+1]!=self.target[1] or not 1<=location<len(route)-2:
                    continue
                # All exceptions are fixed before either boundary is timed.
                before=route[:location+1]
                after=route[location+1:]
                first,pre,failed=self.fixed(before,0.,allocation)
                trial={"mode":self.args.mode,"gate":self.target,"allocation":sorted(allocation),
                       "upstream_depth":len(pre),"prefix_clock":first,"prefix_stall":failed}
                self.cases.append(trial)
                if failed:
                    continue
                options=self.discover(first,self.target in allocation)
                if not options:
                    continue
                for goal in (111.,108.,105.,101.,112.):
                    latest,suffix,fail=self.fixed(after,goal,allocation,True)
                    if fail:
                        continue
                    gap=latest-min(map(arrival,options))
                    self.traces.append({"goal":goal,"gate_slack":gap,"available_release":first,
                                        "earliest_gate_arrival":min(map(arrival,options)),
                                        "latest_gate_arrival":latest})
                    for jump in options:
                        if arrival(jump)<=latest+8e-7:
                            self.submit((*pre,jump,*suffix),trial)
                for jump in options[:3]:
                    _,suffix,fail=self.fixed(after,arrival(jump),allocation)
                    if not fail:
                        self.submit((*pre,jump,*suffix),trial)
                if i%4==0:
                    self.persist()
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.persist()
            print(dict(self.stats),len(self.orders),self.best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("original","alternate"),required=True)
    p.add_argument("--edge",type=int,nargs=2,default=(24,6))
    p.add_argument("--phase",type=float,default=.013)
    p.add_argument("--seed",type=int,default=88101)
    p.add_argument("--seconds",type=float,default=85.)
    p.add_argument("--proposals",type=int,default=25)
    p.add_argument("--windows",type=int,default=6)
    args=p.parse_args()
    args.atlas=FILES/"small_v4/diagnostic/window_e3e82fc984_checkpoint.json"
    args.resume=None
    Benchmark(args).run()
