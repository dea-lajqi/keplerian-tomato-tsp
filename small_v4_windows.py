"""Discover physical Small terminal windows before selecting an upstream tour."""
import argparse
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
import json
import random
import signal
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from continuous_windows_easy import FILES, SEED, WindowSearch
from elite_easy import anchor_bank, scan_archive
from focus_easy import retime
from global_dp_easy import Grid, evaluator
from global_event_easy import Topology, geometry
from kttsp_esa_solver import chromosome, validate
from pareto_easy import BudgetExpired, arrival, write_json
from small_v2 import atomic_json


class Search:
    cost = WindowSearch.cost

    def __init__(self,args):
        self.args=args
        self.started=time.monotonic()
        self.deadline=self.started+args.seconds
        self.module=evaluator()
        self.udp=self.module.udp
        self.normal=self.udp.dv_threshold
        self.limit=self.udp.dv_threshold_exception
        self.cap=self.udp.n_dv_exception
        self.allowed,floors=geometry(self.udp,self.module.MU_MOON)
        self.archive,_=scan_archive(self.udp)
        self.topology=Topology(self.udp,self.allowed,floors,self.archive)
        self.groups=sorted(tuple(sorted(i for i in g)) for g in self.topology.cuts if len(g)==3)
        if len(self.groups)!=3:
            raise ValueError("Expected three Small terminal groups")
        self.high=[i for i,a in enumerate(self.udp.tomato_orbital_parameters[:,0]) if a>1e7]
        self.anchors=anchor_bank(self.archive)
        baseline=json.loads(SEED.read_text())
        self.baseline_path=tuple(baseline["path"])
        self.validation=validate(self.udp,baseline["path"],baseline["transfers"])
        if not self.validation or not self.validation["valid"]:
            raise ValueError("ESA rejected incumbent")
        self.best=self.validation["fitness"][0]
        self.best_generated=None
        self.stats=Counter()
        self.rng=random.Random(args.seed)
        self.atlas=defaultdict(dict)
        self.terminal={}
        self.orders=set()
        self.finishes=set()
        self.attempts=[]
        self.seen=set()
        self.grids={}
        self.output=args.output
        self.output.mkdir(parents=True,exist_ok=True)
        self.job=uuid4().hex[:10]
        self.members={}
        self.topology_seen=set()
        self.processed=set()
        self.discovery_done=set()
        self.atlas_file=self.output/f"window_{self.job}_checkpoint.json"
        self.import_members()
        if args.resume:
            old=json.loads(args.resume.read_text())
            for leg in old["crossings"]:
                self.add(tuple(leg))
            for tail in old["windows"]:
                self.keep_terminal(tuple(map(tuple,tail)))
            self.processed.update((tuple(row[0]),tuple(row[1]),*row[2:])
                                  for row in old.get("processed",()))
            self.topology.forbidden.extend(set(map(tuple,edges)) for edges in old.get("excluded_topologies",()))
            self.discovery_done.update(tuple(row) for row in old.get("discovery_done",()))
            self.topology_seen.update(tuple(row) for row in old.get("topology_seen",()))
            for row in old.get("complete_candidates",()):
                verified=validate(self.udp,row["path"],row["transfers"])
                if verified and verified["valid"] and verified["fitness"][0]<self.best:
                    self.best=verified["fitness"][0]
                    self.best_generated=row
            state=old.get("rng")
            if state:
                self.rng.setstate((state[0],tuple(state[1]),state[2]))

    def import_members(self):
        candidates=[(path,min(rows,key=lambda x:x[0])[1]) for path,rows in self.archive.items()]
        for p in (FILES/"small_v2/campaign_82402").glob("episode_*/*checkpoint.json"):
            candidates.extend((tuple(row["path"]),row["transfers"])
                              for row in json.loads(p.read_text())["pool"])
        for path,legs in candidates:
            if path in self.members and arrival(self.members[path][-1])<=arrival(legs[-1]):
                continue
            if not any(set(path[-3:])==set(g) for g in self.groups):
                continue
            result=validate(self.udp,path,legs)
            if result and result["valid"]:
                self.members[path]=tuple((*l[:4],float(self.udp.compute_transfer(*l[:4]))) for l in legs)

    def check(self):
        if time.monotonic()>=self.deadline:
            raise BudgetExpired

    def grid(self,step,phase,goal=None):
        key=(step,phase,goal)
        if key not in self.grids:
            if len(self.grids)>=9:
                self.grids.pop(next(iter(self.grids)))
            g=Grid(self.module,step,phase,self.deadline,finish_day=goal,directional=True)
            g.horizon=min(24.,goal) if goal else 114.
            self.grids[key]=g
        return self.grids[key]

    def add(self,leg):
        # One early arrival and one late release per quarter-day cell.
        a,b,dep,flight,dv=leg
        bucket=self.atlas[a,b]
        for kind,key in (("early",int(dep*4)),("late",int((dep+flight)*4))):
            slot=(kind,key)
            old=bucket.get(slot)
            value=arrival(leg) if kind=="early" else -dep
            if old is None or value<(arrival(old) if kind=="early" else -old[2]):
                bucket[slot]=leg

    def discover(self,stop):
        pairs=[(a,b) for a in self.high for group in self.groups for b in group if self.allowed[1][a,b]]
        self.rng.shuffle(pairs)
        epochs=(90.,95.,99.,103.,107.)
        jobs=[(a,b,day) for day in epochs for a,b in pairs]
        self.rng.shuffle(jobs)
        for a,b,day in jobs:
            if time.monotonic()>=stop:
                break
            if (a,b,day) in self.discovery_done:
                continue
            sobol=qmc.Sobol(2,scramble=True,seed=self.rng.randrange(2**32)).random_base2(self.args.power)
            sobol[:,0]=day+5*sobol[:,0]
            sobol[:,1]=.14+2.8*sobol[:,1]**1.65
            probes=sorted((self.cost(a,b,d,f),float(d),float(f)) for d,f in sobol)
            starts=[]
            for _,d,f in probes:
                if any(abs(d-x)<.25 and abs(f-y)<.19 for x,y in starts):
                    continue
                starts.append((d,f))
                if len(starts)>=3:
                    break
            for d,f in starts:
                bounds=((max(day,d-.38),min(day+5,d+.38)),(max(self.udp.min_tof,f-.31),f+.31))
                fun=lambda xy:self.cost(a,b,*xy)
                fit=minimize(fun,(d,f),bounds=bounds,method="Powell",options={"maxfev":64,"xtol":.0002})
                if fit.fun>self.limit-1e-6:
                    continue
                x,y=map(float,fit.x)
                self.add((a,b,x,y,float(fit.fun)))
                improve=minimize(sum,fit.x,bounds=bounds,method="SLSQP",
                    constraints=[{"type":"ineq","fun":lambda xy:self.limit-2e-6-fun(xy)}],
                    options={"maxiter":18,"ftol":1e-8,"eps":2e-7})
                v=fun(improve.x)
                if v<=self.limit-1e-6:
                    x,y=map(float,improve.x)
                    self.add((a,b,x,y,v))
            self.discovery_done.add((a,b,day))
            self.stats["crossing_cells"]+=1

    def keep_terminal(self,legs):
        path=(legs[0][0],*(l[1] for l in legs))
        count=sum(l[4]>self.normal for l in legs)
        key=(path,count,int(legs[0][2]*4))
        if key not in self.terminal or arrival(legs[-1])<arrival(self.terminal[key][-1]):
            self.terminal[key]=legs

    def build_terminals(self,stop):
        grid=self.grid(.03125,.012)
        crossings=sorted({l for bucket in self.atlas.values() for l in bucket.values()},
                         key=lambda l:l[2])
        for start in crossings:
            if time.monotonic()>=stop:
                return
            self.check()
            b=start[1]
            group=next(g for g in self.groups if b in g)
            for rest in permutations(i for i in group if i!=b):
                states={int(start[4]>self.normal):(arrival(start),(start,))}
                for a,c in zip((b,*rest),rest):
                    nxt={}
                    for spent,(clock,path) in states.items():
                        for leg in grid.edges(a,c,clock,1.4,1.7,spent<2):
                            if leg is None or leg[3]<self.udp.min_tof:
                                continue
                            used=spent+int(leg[4]>self.normal)
                            if used<=2 and (used not in nxt or arrival(leg)<nxt[used][0]):
                                nxt[used]=(arrival(leg),(*path,leg))
                    states=nxt
                for _,tail in states.values():
                    self.keep_terminal(tail)

    def submit(self,legs,trial):
        key=tuple(l[:4] for l in legs)
        if key in self.seen:
            return
        self.seen.add(key)
        path=(legs[0][0],*(l[1] for l in legs))
        self.stats["esa_checks"]+=1
        valid=validate(self.udp,path,legs)
        if not valid or not valid["valid"]:
            self.stats["esa_rejected"]+=1
            return
        days=valid["fitness"][0]
        self.stats["esa_feasible"]+=1
        self.orders.add(path)
        self.finishes.add(round(days,7))
        trial["valid_days"]=min(days,trial.get("valid_days",1e6))
        entry={"path":path,"transfers":legs,"validation":valid,
               "chromosome":chromosome(path,legs),"trial":dict(trial),"seed":self.args.seed}
        if self.best_generated is None or days<self.best_generated["validation"]["fitness"][0]:
            self.best_generated=entry
        if days<self.best:
            second=validate(evaluator().udp,path,legs)
            if not second or not second["valid"] or second["fitness"][0]>=self.best:
                return
            self.best=second["fitness"][0]
            self.stats["records"]+=1
            write_json(self.output/f"easy_v4_{self.job}_{self.stats['records']:04}.json",entry)

    def join(self,tail,route,legs,count,trial):
        """Backward latest-release DP over the unvisited upstream subset."""
        prefix_len=len(route)-3
        old_prefix=route[:prefix_len]
        source=tail[0][0]
        cut=prefix_len-count-1
        if cut<1 or source not in old_prefix[cut+1:]:
            self.stats["source_outside_neighborhood"]+=1
            return
        prefix=legs[:cut]
        boundary=arrival(prefix[-1])
        due=tail[0][2]
        available=self.cap-sum(l[4]>self.normal for l in (*prefix,*tail))
        if available<0:
            self.stats["exception_budget_rejections"]+=1
            return
        if due-boundary<count*self.udp.min_tof:
            self.stats["chronology_rejections"]+=1
            return
        nodes=tuple(x for x in old_prefix[cut+1:] if x!=source)
        grid=self.grid(.0625,self.args.phase,112.)
        paths={(0,source,0):(due,())}
        trial["prefix_release"]=boundary
        for layer in range(len(nodes)+1):
            nxt={}
            self.stats["join_labels"]+=len(paths)
            for (mask,b,used),(clock,post) in paths.items():
                choices=[(j,a) for j,a in enumerate(nodes) if not mask&(1<<j)] if layer<len(nodes) else [(-1,old_prefix[cut])]
                for j,a in choices:
                    if not self.allowed[1][a,b] or (used==available and not self.allowed[0][a,b]):
                        continue
                    options=[l for l in self.anchors.get((a,b),()) if arrival(l)<=clock+8e-7
                             and l[2]>=boundary-8e-7]
                    raw=grid.edges(b,a,112.-clock,2.2,8.5,used<available)
                    for l in raw:
                        if l is not None and l[3]>=self.udp.min_tof:
                            d=112.-arrival(l)
                            if d>=boundary-8e-7 and d+l[3]<=clock+8e-7:
                                dv=self.cost(a,b,d,l[3])
                                if dv<=self.limit+1e-6:
                                    options.append((a,b,d,l[3],dv))
                    for l in options:
                        resource=used+int(l[4]>self.normal)
                        if resource>available:
                            continue
                        if j<0:
                            self.submit((*prefix,l,*post,*tail),trial)
                        else:
                            key=(mask|1<<j,a,resource)
                            if key not in nxt or l[2]>nxt[key][0]:
                                nxt[key]=(l[2],(l,*post))
            if not nxt:
                break
            if len(nxt)>self.args.beam:
                nxt=dict(sorted(nxt.items(),key=lambda row:(-row[1][0]+.22*row[0][2],
                                                            row[0]))[:self.args.beam])
            paths=nxt

    def route_to_terminal(self,tail,goal,trial):
        """Force the measured terminal arcs into a new route, then time its prefix."""
        path=(tail[0][0],*(l[1] for l in tail))
        used=sum(l[4]>self.normal for l in tail)
        key=(*path,round(tail[0][2],2))
        if key in self.topology_seen:
            return
        self.topology_seen.add(key)
        bound=min(self.deadline,time.monotonic()+4.5)
        ref=set(zip(self.baseline_path,self.baseline_path[1:]))
        result=self.topology.propose(self.rng,bound,ref,5,1.9,
                         forced=(*zip(path,path[1:]),(path[-1],self.udp.n_tomatoes)))
        self.stats["terminal_topology_attempts"]+=1
        if not result or tuple(result[0][-4:])!=path:
            return
        self.stats["terminal_topologies"]+=1
        route=result[0][:-3]
        budget=self.cap-used
        necessary=sum(not self.allowed[0][a,b] for a,b in zip(route,route[1:]))
        trial["mandatory_prefix_exceptions"]=necessary
        trial["available_prefix_exceptions"]=budget
        if necessary>budget:
            self.stats["topology_budget_rejections"]+=1
            return
        due=tail[0][2]
        # Separate grid horizons are essential. The clock must be in the
        # same absolute reference as the terminal opportunity.
        pref=Grid(self.module,.0625,.016,self.deadline,directional=True)
        pref.horizon=due
        schedules,depth=retime(pref,route,self.anchors,wait=4.5,flight=9.5,
                               max_exceptions=budget)
        trial["new_prefix_depth"]=depth
        for schedule in schedules:
            if arrival(schedule[-1])<=due+8e-7:
                self.submit((*schedule,*tail),trial)

    def direct_joins(self):
        by_source=defaultdict(list)
        for tail in self.terminal.values():
            by_source[tail[0][0],tuple(sorted(l[1] for l in tail))].append(tail)
        for route,legs in self.members.items():
            prefix=legs[:-3]
            release=arrival(prefix[-1])
            remaining=self.cap-sum(l[4]>self.normal for l in prefix)
            for tail in by_source[route[-4],tuple(sorted(route[-3:]))]:
                if tail[0][2]+8e-7<release or sum(l[4]>self.normal for l in tail)>remaining:
                    continue
                if arrival(tail[-1])>self.best+.4:
                    continue
                self.stats["direct_joins"]+=1
                trial={"method":"verified prefix join","goal":112.,"crossing":tail[0][:2]}
                self.submit((*prefix,*tail),trial)

    def reconstruct(self,stop):
        groups=defaultdict(list)
        for route,legs in sorted(self.members.items(),key=lambda row:arrival(row[1][-1])):
            groups[tuple(sorted(route[-3:]))].append((route,legs))
        for goal in (110.999,108.,105.,101.,112.):
            local_stop=min(stop,time.monotonic()+max(2.,(stop-time.monotonic())/3))
            tails=[l for l in self.terminal.values() if arrival(l[-1])<=goal]
            tails.sort(key=lambda l:(sum(x[4]>self.normal for x in l),-l[0][2],arrival(l[-1])))
            self.stats[f"deadline_{goal}_windows"]=len(tails)
            count=0
            if self.args.topology:
                # Probe different terminal groups before spending the budget
                # on repeated sources from the incumbent's group.
                for group in self.groups:
                    if time.monotonic()>=local_stop:
                        break
                    picks=[t for t in tails if tuple(sorted(l[1] for l in t))==group
                           and (*((t[0][0],*(l[1] for l in t))),round(t[0][2],2)) not in self.topology_seen]
                    chosen=set()
                    for tail in picks:
                        if len(chosen)>=self.args.topology_per_group or time.monotonic()>=local_stop:
                            break
                        source=tail[0][0]
                        if source in chosen:
                            continue
                        chosen.add(source)
                        trial={"goal":goal,"crossing":tail[0][:2],"finish":arrival(tail[-1]),
                               "departure":tail[0][2],"group":group,"method":"terminal-first topology"}
                        self.attempts.append(trial)
                        self.route_to_terminal(tail,goal,trial)
            for tail in tails:
                if time.monotonic()>=local_stop or count>=self.args.joins:
                    break
                group=tuple(sorted(l[1] for l in tail))
                for route,legs in groups[group][:10]:
                    width=next((n for n in self.args.widths if tail[0][0] in route[-(n+3):-3]),0)
                    if not width:
                        continue
                    if tail[0][0] not in route[-(width+3):-3]:
                        continue
                    cut=len(route)-4-width
                    if sum(l[4]>self.normal for l in (*legs[:cut],*tail))>self.cap:
                        self.stats["exception_budget_rejections"]+=1
                        continue
                    key=(route[-14:],tuple(l[1] for l in tail),round(tail[0][2],4),
                         round(arrival(legs[-11]),4),width,self.args.phase)
                    if key in self.processed:
                        continue
                    self.processed.add(key)
                    trial={"goal":goal,"crossing":tail[0][:2],"finish":arrival(tail[-1]),
                           "departure":tail[0][2],"group":group}
                    self.attempts.append(trial)
                    self.join(tail,route,legs,width,trial)
                    count+=1
                    if count>=self.args.joins or time.monotonic()>=local_stop:
                        break

            self.save()

    def save(self):
        crosses=sorted({leg for bucket in self.atlas.values() for leg in bucket.values()})
        windows=list(self.terminal.values())
        report={"baseline":self.validation,"best_days":self.best,"best_generated":self.best_generated,
                 "elapsed":time.monotonic()-self.started,"seed":self.args.seed,"rng":self.rng.getstate(),
                 "counts":dict(self.stats),"crossings":crosses,"windows":windows,
                 "orders":sorted(self.orders),"finish_bins":sorted(self.finishes),
                 "attempts":self.attempts,"member_groups":dict(Counter(
                     ",".join(map(str,sorted(p[-3:]))) for p in self.members)),
                 "complete_candidates":[self.best_generated] if self.best_generated else [],
                 "excluded_topologies":[sorted(e) for e in self.topology.forbidden],
                 "topology_seen":sorted(self.topology_seen),"processed":list(self.processed),"discovery_done":sorted(self.discovery_done)}
        atomic_json(self.atlas_file,report)

    def run(self):
        try:
            for legs in self.members.values():
                if legs[-3][0] in self.high:
                    self.add(legs[-3])
                    self.keep_terminal(legs[-3:])
            if self.args.mode!="join":
                self.discover(self.started+self.args.seconds*.52)
            self.save()
            self.build_terminals(self.started+self.args.seconds*.78)
            self.save()
            self.direct_joins()
            for _ in range(self.args.rounds):
                self.check()
                self.reconstruct(self.deadline)
                self.save()
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print(self.atlas_file,dict(self.stats),len(self.terminal),self.best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("all","join"),default="all")
    p.add_argument("--seconds",type=float,default=130.)
    p.add_argument("--power",type=int,default=7)
    p.add_argument("--joins",type=int,default=20)
    p.add_argument("--seed",type=int,default=84101)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--widths",type=int,nargs="+",default=[8,10])
    p.add_argument("--phase",type=float,default=.019)
    p.add_argument("--beam",type=int,default=110)
    p.add_argument("--rounds",type=int,default=1)
    p.add_argument("--topology-per-group",type=int,default=1)
    p.add_argument("--topology",action="store_true")
    p.add_argument("--resume",type=Path)
    args=p.parse_args()
    if args.rounds<1 or args.topology_per_group<1 or args.beam<1 or not all(2<=w<=13 for w in args.widths):
        p.error("rounds and topology-per-group must be positive")
    opt=Search(args)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    opt.run()
