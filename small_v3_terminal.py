"""Deadline-first terminal subset DP and continuous terminal basin search."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import permutations
import json
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np

from continuous_windows_easy import FILES, SEED
from global_dp_easy import Grid, evaluator
from focus_easy import retime
from global_event_easy import geometry
from kttsp_esa_solver import chromosome, validate
from paired_crossings_easy import Search as BlockSearch
from pareto_easy import BudgetExpired, arrival, write_json
from small_v2 import atomic_json


class Search:
    cost = BlockSearch.cost
    fit = BlockSearch.fit

    def __init__(self,args):
        self.args=args
        self.started=time.monotonic()
        self.deadline=self.started+args.seconds
        self.module=evaluator()
        self.udp=self.module.udp
        self.rng=random.Random(args.seed)
        self.allowed,_=geometry(self.udp,self.module.MU_MOON)
        self.stats=Counter()
        self.normal=self.udp.dv_threshold
        self.limit=self.udp.dv_threshold_exception+1e-6
        self.cap=self.udp.n_dv_exception
        self.best=json.loads(SEED.read_text())
        checked=validate(self.udp,self.best["path"],self.best["transfers"])
        if not checked or not checked["valid"]:
            raise ValueError("ESA rejected baseline")
        self.best_days=checked["fitness"][0]
        self.incumbent=self.best_days
        self.pool={}
        self.retained={}
        self.load(self.best)
        for f in sorted((FILES/"small_v2/campaign_82402").glob("episode_*/*checkpoint.json")):
            for row in json.loads(f.read_text())["pool"]:
                path=tuple(row["path"])
                legs=row["transfers"]
                days=arrival(legs[-1])
                if path not in self.retained or days<self.retained[path][0]:
                    self.retained[path]=(days,row)
        for _,row in sorted(self.retained.values(),key=lambda item:item[0])[:36]:
            self.load(row)
        self.seen=set()
        self.gate_cache=set()
        self.orders=set()
        self.finishes=set()
        self.families=Counter()
        self.attempts=[]
        self.best_generated=None
        self.best_material_novel=None
        self.job=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
        self.out=args.output
        self.out.mkdir(parents=True,exist_ok=True)
        self.iteration=0
        if args.resume:
            old=json.loads(args.resume.read_text())
            for candidate in old["pool"]:
                self.load(candidate)
            self.iteration=old["next_iteration"]
            state=old["rng"]
            self.rng.setstate((state[0],tuple(state[1]),state[2]))

    def load(self,row):
        path=tuple(row["path"])
        verified=validate(self.udp,path,row["transfers"])
        if verified and verified["valid"]:
            days=verified["fitness"][0]
            if path not in self.pool or days<self.pool[path][0]:
                self.pool[path]=(days,tuple((*l[:4],float(self.udp.compute_transfer(*l[:4])))
                                           for l in row["transfers"]))
            if days<self.best_days:
                self.best_days=days
                self.best=row

    def check(self):
        if time.monotonic()>=self.deadline:
            raise BudgetExpired

    def submit(self,legs,trial):
        key=tuple(l[:4] for l in legs)
        if key in self.seen:
            return
        self.seen.add(key)
        path=(legs[0][0],*(l[1] for l in legs))
        self.stats["checks"]+=1
        checked=validate(self.udp,path,legs)
        if not checked or not checked["valid"]:
            self.stats["rejected"]+=1
            return
        self.stats["feasible"]+=1
        self.orders.add(path)
        days=checked["fitness"][0]
        self.finishes.add(round(days,7))
        self.families[path[-4:]]+=1
        trial["best_valid"]=min(days,trial.get("best_valid",1e6))
        entry={"path":path,"transfers":legs,"validation":checked,
               "chromosome":chromosome(path,legs),"mode":self.args.mode,"trial":dict(trial),
               "seed":self.args.seed}
        if self.best_generated is None or days<self.best_generated["validation"]["fitness"][0]:
            self.best_generated=entry
        if path not in self.retained and (self.best_material_novel is None or
                days<self.best_material_novel["validation"]["fitness"][0]):
            self.best_material_novel=entry
        if path not in self.pool or days<self.pool[path][0]:
            self.pool[path]=(days,legs)
        if days<self.best_days:
            self.best_days=days
            self.best=entry
            self.stats["records"]+=1
            write_json(self.out/f"easy_v3_{self.job}_{self.stats['records']:04}.json",entry)
            print("ESA record",days,flush=True)

    def backward(self,path,legs,left,goal,step,phase,trial):
        """Finite-opportunity subset DP with latest start per exact exception count."""
        source=path[left]
        nodes=path[left+1:]
        full=(1<<len(nodes))-1
        boundary=arrival(legs[left-1]) if left else 0.
        prefix=legs[:left]
        available=self.cap-sum(l[4]>self.normal for l in prefix)
        original_boundary=boundary
        if self.args.mode=="gate":
            boundary=max(0.,boundary-(.6,1.4,2.5)[self.iteration%3])
            trial["release_advance_limit"]=original_boundary-boundary
        horizon=min(goal,max(1.,goal-boundary+step))
        if goal-boundary<len(nodes)*self.udp.min_tof-1e-6:
            return
        grid=Grid(self.module,step,phase,self.deadline,finish_day=goal,directional=True)
        grid.horizon=horizon
        exact={}
        def options(a,b,clock,extra):
            opts=[]
            for l in exact.get((a,b),()):
                if arrival(l)<=clock+8e-7 and (extra or l[4]<=self.normal):
                    opts.append(l)
            release=max(0.,goal-clock)
            for l in grid.edges(b,a,release,1.8 if len(nodes)>7 else 2.8,8.,extra):
                if l is None or l[3]<self.udp.min_tof:
                    continue
                d=goal-arrival(l)
                if d<boundary-1e-6 or d+l[3]>clock+8e-7:
                    continue
                dv=self.cost(a,b,d,l[3])
                if dv<=(self.limit if extra else self.normal):
                    opts.append((a,b,d,l[3],dv))
            return opts
        for tour in (legs,*(row[1] for row in list(self.pool.values())[:12])):
            for l in tour[left-4 if left>=4 else 0:]:
                if l[2]>=boundary-1e-6 and arrival(l)<=goal+8e-7:
                    exact.setdefault(l[:2],[]).append(l)
        states={(1<<j,b,0):(goal,()) for j,b in enumerate(nodes)}
        for _ in range(len(nodes)-1):
            self.check()
            nxt={}
            self.stats["dp_labels"]+=len(states)
            for (mask,b,e),(clock,post) in states.items():
                for j,a in enumerate(nodes):
                    if mask&(1<<j) or not self.allowed[1][a,b]:
                        continue
                    if e>=available and not self.allowed[0][a,b]:
                        continue
                    for l in options(a,b,clock,e<available):
                        use=e+int(l[4]>self.normal)
                        key=(mask|(1<<j),a,use)
                        if use<=available and (key not in nxt or l[2]>nxt[key][0]):
                            nxt[key]=(l[2],(l,*post))
            states=nxt
            trial["tail_visits"]=max(trial.get("tail_visits",0),_+2 if states else _+1)
            if not states:
                return
        for (mask,b,e),(clock,tail) in states.items():
            if mask!=full or not self.allowed[1][source,b]:
                continue
            for leg in options(source,b,clock,e<available):
                if leg[2]+8e-7<boundary or e+int(leg[4]>self.normal)>available:
                    continue
                self.stats["closed_tails"]+=1
                if leg[2]+8e-7>=original_boundary:
                    self.submit((*prefix,leg,*tail),trial)
                elif self.args.mode=="gate":
                    need=original_boundary-leg[2]
                    trial["smallest_advance_needed"]=min(need,trial.get("smallest_advance_needed",1e6))
                    self.gate_repair(path,legs,left,(leg,*tail),trial)

    def gate_repair(self,path,legs,left,tail,trial):
        """Use a measured latest release as an explicit upstream repair target."""
        signature=(tuple(path[:left+1]),round(tail[0][2],5),sum(l[4]>self.normal for l in tail))
        if signature in self.gate_cache:
            return
        self.gate_cache.add(signature)
        self.stats["early_terminal_gates"]+=1
        due=tail[0][2]
        budget=self.cap-signature[2]
        anchors={}
        for l in legs:
            anchors.setdefault(l[:2],[]).append(l)
        first=max(0,left-(7,13,20)[self.iteration%3])
        grid=Grid(self.module,.03125,(self.iteration%4)*.006,self.deadline,directional=True,refine=True)
        grid.horizon=due
        schedules,depth=retime(grid,path[:left+1],anchors,prefix=legs[:first],
                               wait=3.5,flight=9.,max_exceptions=budget)
        trial["prefix_repair_depth"]=max(depth,trial.get("prefix_repair_depth",0))
        for schedule in schedules:
            if arrival(schedule[-1])<=due+8e-7:
                self.submit((*schedule,*tail),trial)

    def continuous(self,path,legs,left,goal,trial):
        """Reallocate exceptions and fit the whole terminal block at once."""
        prefix=legs[:left]
        tail=legs[left:]
        pairs=tuple(l[:2] for l in tail)
        boundary=arrival(prefix[-1]) if prefix else 0.
        available=self.cap-sum(l[4]>self.normal for l in prefix)
        mandatory={i for i,(a,b) in enumerate(pairs) if not self.allowed[0][a,b]}
        original={i for i,l in enumerate(tail) if l[4]>self.normal}
        allocations=[original, mandatory]
        if len(mandatory)<available:
            allocations.extend(mandatory|{i} for i in range(len(pairs)) if i not in mandatory)
        visited=set()
        for allocation in allocations:
            key=tuple(sorted(allocation))
            if key in visited or len(allocation)>available:
                continue
            visited.add(key)
            limits=np.array([self.limit-3e-7 if i in allocation else self.normal-3e-7
                             for i in range(len(tail))])
            x=np.array([l[2:4] for l in tail])
            delta=(0.,-.09,-.21,.16,-.4,.33)[self.iteration%6]
            x[:,0]+=np.linspace(0.,delta,len(tail))
            x[:,1]=np.maximum(self.udp.min_tof,x[:,1]*(1.+.08*delta))
            seed=self.fit(pairs,x,boundary,goal,limits,1.25,trial)
            if seed:
                self.submit((*prefix,*seed),trial)

    def permutations_continuous(self,path,legs,trial):
        """Enumerate terminal orders and jointly fit each order's absolute clocks."""
        left=len(legs)-3
        prefix=legs[:left]
        release=arrival(prefix[-1])
        available=self.cap-sum(l[4]>self.normal for l in prefix)
        for nodes in permutations(path[left+1:]):
            pairs=tuple(zip((path[left],*nodes),nodes))
            if not all(self.allowed[1][a,b] for a,b in pairs):
                continue
            mandatory={i for i,(a,b) in enumerate(pairs) if not self.allowed[0][a,b]}
            if len(mandatory)>available:
                continue
            limits=np.array([self.limit-2e-7 if i in mandatory else self.normal-2e-7
                             for i in range(len(pairs))])
            self.stats["permutation_trials"]+=1
            old=np.array([l[2:4] for l in legs[left:]])
            for j in range(4):
                x=old.copy()
                if j:
                    x[:,0]+=np.array([self.rng.uniform(-.24,.24) for _ in pairs])
                    x[:,1]=np.maximum(self.udp.min_tof,x[:,1]+np.array(
                                      [self.rng.uniform(-.12,.12) for _ in pairs]))
                block=self.fit(pairs,x,release,trial["goal"],limits,.48,trial)
                if block:
                    self.submit((*prefix,*block),trial)

    def save(self):
        ranked=sorted(self.pool.items(),key=lambda item:item[1][0])[:42]
        atomic_json(self.out/f"v3_{self.job}_checkpoint.json",
                    {"next_iteration":self.iteration,"rng":self.rng.getstate(),
                     "stats":dict(self.stats),"orders":sorted(self.orders),"finish_bins":sorted(self.finishes),
                     "families":[(k,v) for k,v in self.families.items()],"attempts":self.attempts,
                     "best_days":self.best_days,"best_generated":self.best_generated,
                     "best_new_order":self.best_material_novel,"best":self.best,
                     "elapsed":time.monotonic()-self.started,"seed":self.args.seed,
                     "pool":[{"path":p,"transfers":l} for p,(_,l) in ranked]})

    def run(self):
        try:
            while self.best_days>101.:
                self.check()
                pool=sorted(self.pool.items(),key=lambda item:item[1][0])
                i=self.iteration
                k=(i//3)%min(22,len(pool))
                path,(days,legs)=pool[k]
                if self.args.mode in ("backward","gate"):
                    count=(4,6,8,5,7,9)[i%6]
                else:
                    count=(3,5,8,4,6,10)[i%6]
                left=len(legs)-count
                goals=(111.755,111.73,111.68,111.5,111.,110.,108.,112.,113.)
                goal=goals[(i//2+i%2)%len(goals)]
                if self.args.mode=="gate":
                    count=(4,5,6,4,7)[i%5]
                    left=len(legs)-count
                    goal=(111.74,111.5,111.1,110.9,112.,111.68,110.,108.)[i%8]
                if self.args.mode=="continuous":
                    goal=min(goal,days)-.001
                trial={"iteration":i,"parent_finish":days,"goal":goal,"tail_size":count}
                self.attempts.append(trial)
                if self.args.mode in ("backward","gate"):
                    self.backward(path,legs,left,goal,(.0390625,.03125,.0234375)[i%3] if self.args.mode=="gate" else (.015625,.03125,.01171875)[i%3],
                                  (.002,.007,.004)[(i//3)%3],trial)
                elif self.args.mode=="terminal-permutations":
                    self.permutations_continuous(path,legs,trial)
                else:
                    self.continuous(path,legs,left,goal,trial)
                self.iteration+=1
                if i%8==0:
                    self.save()
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print(dict(self.stats),"orders",len(self.orders),"finishes",len(self.finishes),
                  "best",self.best_days,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("backward","continuous","gate","terminal-permutations"),required=True)
    p.add_argument("--seconds",type=float,default=110.)
    p.add_argument("--seed",type=int,default=83101)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",type=Path)
    Search(p.parse_args()).run()
