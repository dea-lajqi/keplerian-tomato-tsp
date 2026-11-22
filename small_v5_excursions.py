"""Small-only search over movable low-shell excursions and timed upstream joins."""
import argparse
from collections import Counter, defaultdict
from copy import copy
from itertools import permutations
import json
from pathlib import Path
import random
import signal
import time
from uuid import uuid4

from continuous_windows_easy import FILES, SEED
from elite_easy import anchor_bank, scan_archive
from focus_easy import retime
from global_dp_easy import Grid, evaluator
from global_event_easy import Topology, geometry
from kttsp_esa_solver import chromosome, validate
from pareto_easy import BudgetExpired, arrival, write_json
from small_v2 import atomic_json


def route_parts(order,groups):
    high=tuple(x for x in order if all(x not in g for g in groups))
    return high


class Search:
    def __init__(self,args):
        self.args=args
        self.start=time.monotonic()
        self.stop=self.start+args.seconds
        self.rng=random.Random(args.seed)
        self.module=evaluator()
        self.udp=self.module.udp
        self.threshold=self.udp.dv_threshold
        self.cap=self.udp.n_dv_exception
        self.geometry,self.floors=geometry(self.udp,self.module.MU_MOON)
        self.normal,self.extra=self.geometry
        self.baseline=json.loads(SEED.read_text())
        self.reference=tuple(self.baseline["path"])
        checked=validate(self.udp,self.reference,self.baseline["transfers"])
        if not checked or not checked["valid"]:
            raise ValueError("ESA rejected incumbent")
        self.best=checked["fitness"][0]
        self.original=self.best
        self.archive,_=scan_archive(self.udp)
        self.anchors=anchor_bank(self.archive)
        v4=json.loads(args.atlas.read_text())
        # Retain one-exception tails to leave room for the other shell transitions.
        self.windows=defaultdict(list)
        for rows in v4["windows"]:
            legs=tuple(map(tuple,rows))
            if sum(l[4]>self.threshold for l in legs)!=1:
                continue
            if any(l[4]>self.udp.dv_threshold_exception+1e-6 for l in legs):
                continue
            self.windows[tuple(sorted(l[1] for l in legs))].append(legs)
        self.groups=sorted(self.windows)
        self.reference_high=route_parts(self.reference,self.groups)
        self.job=uuid4().hex[:10]
        self.output=args.output
        self.output.mkdir(parents=True,exist_ok=True)
        self.checkpoint=self.output/f"v5_{self.job}_checkpoint.json"
        self.stats=Counter()
        self.history=[]
        self.attempted=set()
        self.orders=set()
        self.finishes=set()
        self.seen=set()
        self.best_candidate=None
        self.grids={}
        self.last_saved=self.start
        self.population=set()
        for p in self.archive:
            self.population.add(route_parts(p,self.groups))
        for file in (FILES/"small_v2/campaign_82402").glob("episode_*/*checkpoint.json"):
            for row in json.loads(file.read_text())["pool"]:
                self.population.add(route_parts(row["path"],self.groups))
        if args.resume:
            prev=json.loads(args.resume.read_text())
            self.attempted.update(tuple(x) for x in prev["attempted"])
            row=prev.get("best_candidate")
            if row:
                result=validate(self.udp,row["path"],row["transfers"])
                if result and result["valid"] and result["fitness"][0]<self.best:
                    self.best=result["fitness"][0]
                    self.best_candidate=row

    def grid(self,step,phase,goal=None):
        key=step,phase,goal
        if key not in self.grids:
            if len(self.grids)>=9:
                self.grids.pop(next(iter(self.grids)))
            self.grids[key]=Grid(self.module,step,phase,self.stop,finish_day=goal,directional=True)
            self.grids[key].horizon=114. if goal is None else min(goal,72.)
        return self.grids[key]

    def necessary(self,path):
        return sum(not self.normal[a,b] for a,b in zip(path,path[1:]))

    def skeletons(self):
        if self.args.static_seeds:
            reduced=copy(self.udp)
            reduced.n_dv_exception=4
            topology=Topology(reduced,self.geometry,self.floors,self.archive)
            for i in range(self.args.static_seeds):
                if time.monotonic()+2.>=self.stop:
                    break
                first,last=self.rng.sample(self.groups,2)
                result=topology.propose(self.rng,min(self.stop,time.monotonic()+3.),
                      self.ref_edges,4,1.6,
                      forced=((self.udp.n_tomatoes,self.rng.choice(first)),
                              (self.rng.choice(last),self.udp.n_tomatoes)))
                self.stats["four_exception_static_attempts"]+=1
                if result:
                    self.stats["four_exception_static_paths"]+=1
                    self.population.add(route_parts(result[0],self.groups))
        original=sorted(self.population)
        candidates=set(original)|{p[::-1] for p in original}
        for route in original[:65]:
            for i in range(1,len(route)-3):
                for j in range(i+2,len(route)-1):
                    if not self.extra[route[i-1],route[j]] or not self.extra[route[i],route[j+1]]:
                        continue
                    path=(*route[:i],*route[i:j+1][::-1],*route[j+1:])
                    if self.necessary(path)<=2:
                        candidates.add(path)
        if self.args.rotate:
            for path in list(candidates):
                if not self.extra[path[-1],path[0]]:
                    continue
                for j in range(1,len(path)):
                    p=path[j:]+path[:j]
                    if self.necessary(p)<=2:
                        candidates.add(p)
        ranked=sorted(candidates,key=lambda p:(self.necessary(p),self.rng.random()))
        self.stats["high_shell_skeletons"]=len(ranked)
        return ranked

    def proposals(self,tail_group,source):
        """Compose start group, high-shell order and one movable low excursion."""
        for skeleton in self.skeletons_bank:
            if skeleton[-1]!=source:
                continue
            bad=[i for i,(a,b) in enumerate(zip(skeleton,skeleton[1:])) if not self.normal[a,b]]
            positions=range(len(skeleton)-1)
            if len(bad)>=2:
                positions=bad
            for g in self.groups:
                if g==tail_group:
                    continue
                middle=next(x for x in self.groups if x!=g and x!=tail_group)
                for init in permutations(g):
                    if not self.extra[init[-1],skeleton[0]]:
                        continue
                    for i in positions:
                        remaining=len(bad)-(i in bad)
                        if remaining>(0 if self.args.four_only else 1):
                            continue
                        a,b=skeleton[i:i+2]
                        for mid in permutations(middle):
                            if not self.extra[a,mid[0]] or not self.extra[mid[-1],b]:
                                continue
                            prefix=(*init,*skeleton[:i+1],*mid,*skeleton[i+1:])
                            self.stats["structural_orders"]+=1
                            yield prefix,i+3,remaining

    def prefix_dp(self,route,budget,trial,fine=False):
        step=.03125 if fine else .0625
        grid=self.grid(step,self.args.phase % step)
        schedules,depth=retime(grid,route,self.anchors,wait=4.5,flight=9.5,max_exceptions=budget)
        trial["prefix_depth"]=max(depth,trial.get("prefix_depth",0))
        return schedules

    def reverse_profiles(self,route,deadline,budget):
        base=112.
        grid=self.grid(.0625,self.args.phase,base)
        anchor=defaultdict(list)
        for a,b in zip(route,route[1:]):
            for l in self.anchors.get((a,b),()):
                dep=base-arrival(l)
                if dep>=0:
                    anchor[b,a].append((b,a,dep,l[3],l[4]))
        states={0:(base-deadline,())}
        depth=0
        for b,a in zip(route[:0:-1],route[-2::-1]):
            nxt={}
            for used,(clock,suffix) in states.items():
                opts=(*grid.edges(b,a,clock,3.8,9.5,used<budget),*anchor.get((b,a),()))
                for l in opts:
                    if l is None or l[2]+8e-7<clock or l[3]<self.udp.min_tof:
                        continue
                    d=base-arrival(l)
                    if d<0:
                        continue
                    dv=float(self.udp.compute_transfer(a,b,d,l[3]))
                    e=used+int(dv>self.threshold)
                    if e<=budget and dv<=self.udp.dv_threshold_exception+1e-6 and (
                            e not in nxt or arrival(l)<nxt[e][0]):
                        nxt[e]=(arrival(l),((a,b,d,l[3],dv),*suffix))
            if not nxt:
                return [],depth
            depth+=1
            states=nxt
        return [s for _,s in states.values()],depth

    def submit(self,legs,trial):
        key=tuple(l[:4] for l in legs)
        if key in self.seen:
            return
        self.seen.add(key)
        path=(legs[0][0],*(l[1] for l in legs))
        self.stats["esa_checks"]+=1
        result=validate(self.udp,path,legs)
        if not result or not result["valid"]:
            self.stats["esa_rejected"]+=1
            return
        days=result["fitness"][0]
        self.stats["esa_feasible"]+=1
        self.orders.add(path)
        self.finishes.add(round(days,7))
        trial["valid_days"]=min(days,trial.get("valid_days",1e6))
        row={"path":path,"transfers":legs,"validation":result,"chromosome":chromosome(path,legs),
             "seed":self.args.seed,"experiment":dict(trial)}
        if self.best_candidate is None or days<self.best_candidate["validation"]["fitness"][0]:
            self.best_candidate=row
        if days<self.best:
            independent=validate(evaluator().udp,path,legs)
            if independent and independent["valid"] and independent["fitness"][0]<self.best:
                self.best=independent["fitness"][0]
                self.stats["records"]+=1
                write_json(self.output/f"easy_v5_{self.job}_{self.stats['records']:04}.json",row)

    def attempt(self,route,split,tail,goal,trial):
        budget=self.cap-1
        if self.args.method=="forward":
            schedules=self.prefix_dp(route,budget,trial)
            if self.args.rotate and goal<112 and schedules and (
                    0.<min(arrival(p[-1]) for p in schedules)-tail[0][2]<.8):
                self.stats["fine_gate_retries"]+=1
                schedules += self.prefix_dp(route,budget,trial,True)
            for prefix in schedules:
                wait=tail[0][2]-arrival(prefix[-1])
                trial["best_gap"]=max(wait,trial.get("best_gap",-1e6))
                if wait>=-8e-7:
                    self.submit((*prefix,*tail),trial)
            return
        split=min(len(route)-3,split+3)
        early,late=route[:split+1],route[split:]
        min_early,min_late=self.necessary(early),self.necessary(late)
        left=self.prefix_dp(early,budget-min_late,trial)
        if not left:
            return
        suffix,depth=self.reverse_profiles(late,tail[0][2],budget-min_early)
        trial["backward_depth"]=depth
        for pre in left:
            for post in suffix:
                used=sum(l[4]>self.threshold for l in (*pre,*post))
                gap=post[0][2]-arrival(pre[-1])
                if used<=budget:
                    trial["best_gap"]=max(gap,trial.get("best_gap",-1e6))
                    if gap>=-8e-7:
                        self.submit((*pre,*post,*tail),trial)

    def save(self):
        atomic_json(self.checkpoint,{"best_days":self.best,"best_candidate":self.best_candidate,
             "elapsed":time.monotonic()-self.start,"counts":dict(self.stats),
             "method":self.args.method,"four_only":self.args.four_only,"seed":self.args.seed,
             "attempted":sorted(self.attempted),"orders":sorted(self.orders),
             "finish_bins":sorted(self.finishes),"history":self.history})

    def run(self):
        try:
            self.skeletons_bank=self.skeletons()
            scored={}
            for group in self.groups:
                sources=sorted({l[0][0] for l in self.windows[group]} &
                               {p[-1] for p in self.skeletons_bank})
                for source in sources:
                    rows={}
                    for p,split,remaining in self.proposals(group,source):
                        overlap=sum((a,b) in self.ref_edges for a,b in zip(p,p[1:]))
                        merit=(.7 if self.args.four_only else 2.5)*(len(p)-1-overlap)+self.rng.random()*4.
                        rows[p]=(merit,split,remaining)
                    if rows:
                        near=sorted(rows.items(),key=lambda row:row[1][0])
                        scored[group,source]=near[:40]+self.rng.sample(near[40:],min(15,len(near[40:])))
            self.stats["terminal_sources_with_orders"]=len(scored)
            goals=(110.999,108.,105.,101.,112.)
            choices={}
            for goal in goals:
                for group,source in scored:
                    tails=[l for l in self.windows[group] if l[0][0]==source and arrival(l[-1])<=goal]
                    tails.sort(key=lambda l:(-l[0][2],arrival(l[-1])))
                    unique={}
                    for l in tails:
                        unique.setdefault((int(l[0][2]*4),l[0][1]),l)
                    if unique:
                        choices[goal,group,source]=list(unique.values())[:4]
                self.stats[f"eligible_gate_families_{goal}"]=sum(1 for key in choices if key[0]==goal)
            for cycle in range(self.args.rounds):
                for goal in goals:
                    pairs=[key for key in choices if key[0]==goal]
                    pairs.sort(key=lambda k:(scored[k[1:]][0][1][0],self.rng.random()))
                    if self.args.rotate and len(pairs)>self.args.sources:
                        pick=(cycle*self.args.sources)%len(pairs)
                        pairs=pairs[pick:]+pairs[:pick]
                    for _,group,source in pairs[:self.args.sources]:
                        if time.monotonic()>=self.stop:
                            raise BudgetExpired
                        paths=scored[group,source]
                        route,(_,split,available)=paths[cycle%len(paths)]
                        tails=choices[goal,group,source]
                        tail=tails[(cycle//len(paths))%len(tails)]
                        key=(*route,goal,round(tail[0][2],5))
                        if key in self.attempted:
                            continue
                        self.attempted.add(key)
                        trial={"goal":goal,"group":group,"source":source,"gate":tail[0][2],
                               "terminal_finish":arrival(tail[-1]),"high_bridge_exceptions":available,
                               "changed_arcs":sum(pair not in self.ref_edges for pair in zip(route,route[1:]))}
                        self.history.append(trial)
                        self.attempt(route,split,tail,goal,trial)
                        if time.monotonic()-self.last_saved>20.:
                            self.save()
                            self.last_saved=time.monotonic()
                if self.best<=101.:
                    break
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print(self.checkpoint,dict(self.stats),"best",self.best,flush=True)

    @property
    def ref_edges(self):
        if not hasattr(self,"_ref_edges"):
            self._ref_edges=set(zip(self.reference,self.reference[1:]))
        return self._ref_edges


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--seconds",type=float,default=100.)
    parser.add_argument("--rounds",type=int,default=25)
    parser.add_argument("--sources",type=int,default=5)
    parser.add_argument("--seed",type=int,default=85101)
    parser.add_argument("--phase",type=float,default=.017)
    parser.add_argument("--rotate",action="store_true")
    parser.add_argument("--static-seeds",type=int,default=0)
    parser.add_argument("--four-only",action="store_true")
    parser.add_argument("--method",choices=("forward","meet"),default="meet")
    search=Search(parser.parse_args())
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    search.run()
