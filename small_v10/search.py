"""Whole-tour topology discovery with measured timing feedback, ESA Small only."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import signal
import sys
import time
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from continuous_windows_easy import FILES, SEED
from elite_easy import anchor_bank, anchored_retime, scan_archive
from focus_easy import retime, refine_joint
from global_dp_easy import Grid, evaluator
from global_event_easy import Topology, geometry
from kttsp_esa_solver import chromosome, validate
from pareto_easy import BudgetExpired, arrival, write_json
from small_v2 import atomic_json


class Search:
    def __init__(self,args):
        self.args=args
        self.started=time.monotonic()
        self.stop=self.started+args.seconds
        self.rng=random.Random(args.seed)
        self.module=evaluator()
        self.udp=self.module.udp
        self.allowed,self.floor=geometry(self.udp,self.module.MU_MOON)
        self.archive,_=scan_archive(self.udp)
        self.topology=Topology(self.udp,self.allowed,self.floor,self.archive)
        self.anchors=anchor_bank(self.archive)
        self.incumbent=json.loads(SEED.read_text())
        self.reference=tuple(self.incumbent["path"])
        self.original_edges=set(zip(self.reference,self.reference[1:]))
        check=validate(self.udp,self.reference,self.incumbent["transfers"])
        if not check or not check["valid"]:
            raise ValueError("ESA rejected the fallback incumbent")
        self.best=check["fitness"][0]
        self.base=check
        self.best_new=None
        self.best_generated=None
        self.stats=Counter()
        self.orders=set()
        self.novel=set()
        self.seen=set()
        self.attempted=set()
        self.finish_bins=set()
        self.grids={}
        self.hist=[]
        self.pool={p:min(rows,key=lambda r:r[0])[:2] for p,rows in self.archive.items()}
        self.shell={i:0 for i in range(self.udp.n_tomatoes)}
        for k,g in enumerate(sorted(tuple(sorted(g)) for g in self.topology.cuts if len(g)==3),1):
            self.shell.update({i:k for i in g})
        self.out=args.output
        self.out.mkdir(parents=True,exist_ok=True)
        self.job=uuid4().hex[:10]
        self.checkpoint=self.out/f"global_{self.job}_checkpoint.json"
        self.iteration=0
        self.last_saved=self.started
        self.learned_events={}
        if args.resume:
            old=json.loads(args.resume.read_text())
            self.iteration=old["next_iteration"]
            self.orders.update(map(tuple,old.get("orders",())))
            self.novel.update(map(tuple,old.get("novel_orders",())))
            self.attempted.update(map(tuple,old.get("attempted",())))
            self.finish_bins.update(old.get("finish_bins",()))
            self.topology.failed.update({tuple(edge):n for edge,n in old.get("failed_arcs",())})
            self.topology.forbidden.extend(set(map(tuple,e)) for e in old.get("excluded_topologies",()))
            state=old["rng"]
            self.rng.setstate((state[0],tuple(state[1]),state[2]))
            for row in old.get("pool",()):
                verified=validate(self.udp,row["path"],row["transfers"])
                if verified and verified["valid"]:
                    self.pool[tuple(row["path"])]=(verified["fitness"][0],tuple(map(tuple,row["transfers"])))
                    self.best=min(self.best,verified["fitness"][0])
            for row in old.get("learned_events",()):
                self.remember(tuple(row))
            for name in ("best_new","best_generated"):
                row=old.get(name)
                if row and validate(self.udp,row["path"],row["transfers"])["valid"]:
                    setattr(self,name,row)

    def check(self):
        if time.monotonic()>=self.stop:
            raise BudgetExpired

    def grid(self,step,phase,fine=False):
        key=step,phase,fine
        if key not in self.grids:
            self.grids[key]=Grid(self.module,step,phase,self.stop,refine=fine,directional=True)
            self.grids[key].horizon=min(self.udp.max_time,132.)
        return self.grids[key]

    def family(self,route):
        result=[]
        for a in route:
            group=self.shell[a]
            if not result or group!=result[-1]:
                result.append(group)
        return tuple(result)

    def submit(self,path,legs,trial):
        legs=tuple(map(tuple,legs))
        sig=tuple(l[:4] for l in legs)
        if sig in self.seen:
            return
        self.seen.add(sig)
        path=tuple(path)
        result=validate(self.udp,path,legs)
        self.stats["esa_checks"]+=1
        if not result or not result["valid"]:
            self.stats["esa_rejected"]+=1
            return
        self.stats["esa_feasible"]+=1
        self.orders.add(path)
        days=result["fitness"][0]
        self.finish_bins.add(round(days,7))
        trial["valid_days"]=min(days,trial.get("valid_days",1e6))
        edges=set(zip(path,path[1:]))
        difference=len(edges-self.original_edges)
        common=next((i for i,(a,b) in enumerate(zip(path,self.reference)) if a!=b),len(path))
        record={"path":path,"transfers":legs,"validation":result,"chromosome":chromosome(path,legs),
                 "seed":self.args.seed,"changed_arcs":difference,"shared_initial_visits":common,
                 "family":self.family(path),"experiment":dict(trial)}
        novel=path not in self.archive
        if novel:
            self.novel.add(path)
            if self.best_new is None or days<self.best_new["validation"]["fitness"][0]:
                self.best_new=record
        if self.best_generated is None or days<self.best_generated["validation"]["fitness"][0]:
            self.best_generated=record
        old=self.pool.get(path)
        if old is None or days<old[0]:
            self.pool[path]=(days,tuple(legs))
        if self.args.learn:
            self.topology.observe(legs)
            for l in legs:
                if l not in self.anchors.get(l[:2],()):
                    self.anchors[l[:2]]=(*self.anchors.get(l[:2],()),l)
        if days<self.best:
            verified=validate(evaluator().udp,path,legs)
            if verified and verified["valid"] and verified["fitness"][0]<self.best:
                self.best=verified["fitness"][0]
                self.stats["records"]+=1
                write_json(self.out/f"easy_v10_{self.job}_{self.stats['records']:04}.json",record)
                print("independent ESA record",self.best,flush=True)

    def reproduce(self):
        # Independently validate the stored vector and the decoder's
        # chronological replay from its measured opportunity entries.
        exact={tuple(l[:2]):(tuple(l),) for l in self.incumbent["transfers"]}
        schedules=anchored_retime(self.udp,self.reference,exact)
        trial={"mode":"incumbent replay","depth":len(self.reference)-1 if schedules else 0}
        self.stats["incumbent_replayed"]=int(bool(schedules))
        self.hist.append(trial)
        self.submit(self.reference,self.incumbent["transfers"],trial)
        for schedule in schedules:
            self.submit(self.reference,schedule,trial)

    def parent(self):
        ranked=sorted(self.pool.items(),key=lambda row:row[1][0])
        pool=ranked[:min(len(ranked),6 if self.iteration%3 else 30)]
        return self.rng.choice(pool)[0]

    def proposal(self):
        # Every solve has free initial and terminal nodes. Different amounts
        # of directed-edge overlap enforce different global neighborhoods.
        k=self.iteration
        reference=self.parent() if k%4==3 else self.reference
        target=set(zip(reference,reference[1:]))
        lower=(5,9,15,22,30,37)[k%6]
        # A band prevents spending all solves on the same nearby family.
        upper=min(48,lower+(8 if k%2 else 14))
        return self.topology.propose(self.rng,min(self.stop,time.monotonic()+5.),
                       target,lower,(.8,1.5,2.45)[k%3],maximum_difference=upper)

    def remember(self,leg):
        a,b,d,f,dv=leg
        key=(a,b,round(d*4),int(dv>self.udp.dv_threshold))
        before=self.learned_events.get(key)
        if before and arrival(before)<=arrival(leg):
            return
        existing=self.anchors.get((a,b),())
        if before:
            existing=tuple(row for row in existing if row!=before)
        self.learned_events[key]=leg
        self.anchors[a,b]=(*existing,leg) if leg not in existing else existing
        self.topology.observe((leg,))
        self.stats["partial_events_updated"]+=1

    def run_route(self,route,trial):
        depth_best=0
        options=()
        phases=((.0625,.011),(.046875,.022),(.0625,.041),(.03125,.008))
        for step,phase in (phases[self.iteration%4],phases[(self.iteration+1)%4]):
            grid=self.grid(step,phase)
            timed,depth=retime(grid,route,self.anchors,wait=5.25,flight=10.,
                              observer=self.remember if self.args.partial_feedback else None)
            depth_best=max(depth,depth_best)
            if timed:
                options=timed
                break
        trial["depth"]=depth_best
        if not options and self.args.learn and depth_best<len(route)-1:
            pair=route[depth_best:depth_best+2]
            # A failed phase sample is soft evidence. It never deletes arcs.
            self.topology.failed[pair]+=1
            self.stats["timing_feedback_updates"]+=1
        for schedule in options:
            self.submit(route,schedule,trial)
        if self.args.reverse:
            self.backward_timing(route,trial,options)
        if not options:
            return
        first=min(options,key=lambda s:arrival(s[-1]))
        if self.args.learn and (self.iteration%4==0 or arrival(first[-1])<self.best+.20):
            refine_joint(self.udp,first,self.stop,lambda s:self.submit(route,s,trial),
                         0.,self.rng,False)

    def backward_timing(self,route,trial,forward_options):
        goals=(111.75,114.,117.,121.5,126.)
        goal=goals[self.iteration%len(goals)]
        if forward_options:
            goal=min(goal,min(arrival(legs[-1]) for legs in forward_options)+.10)
        reverse=Grid(self.module,.0625,.019,self.stop,finish_day=goal,directional=True)
        reverse.z_speed=-reverse.z_speed
        anchor=defaultdict(list)
        for a,b in zip(route,route[1:]):
            for l in self.anchors.get((a,b),()):
                dep=goal-arrival(l)
                if dep>=0:
                    anchor[b,a].append((b,a,dep,l[3],l[4]))
        profiles,depth=retime(reverse,route[::-1],anchor,wait=4.,flight=9.5)
        trial["reverse_goal"]=goal
        trial["reverse_depth"]=depth
        self.stats["reverse_attempts"]+=1
        for profile in profiles:
            legs=[]
            for b,a,d,f,_ in reversed(profile):
                dep=goal-d-f
                if dep<0.:
                    break
                dv=float(self.udp.compute_transfer(a,b,dep,f))
                if dv>self.udp.dv_threshold_exception+1e-6:
                    break
                legs.append((a,b,dep,f,dv))
            else:
                self.submit(route,tuple(legs),trial)

    def save(self):
        pool=sorted(self.pool.items(),key=lambda row:row[1][0])[:45]
        self.checkpoint=self.out/f"global_{self.job}_{self.iteration:04d}_checkpoint.json"
        atomic_json(self.checkpoint,{"baseline":self.base,"best_days":self.best,
             "best_generated":self.best_generated,"best_new":self.best_new,
             "elapsed":time.monotonic()-self.started,"counts":dict(self.stats),
             "attempted":sorted(self.attempted),"orders":sorted(self.orders),
             "novel_orders":sorted(self.novel),"finish_bins":sorted(self.finish_bins),
             "history":self.hist,"next_iteration":self.iteration,"seed":self.args.seed,
             "rng":self.rng.getstate(),"failed_arcs":list(self.topology.failed.items()),
             "excluded_topologies":[sorted(x) for x in self.topology.forbidden],
             "learned_events":list(self.learned_events.values()),
             "pool":[{"path":p,"transfers":l} for p,(_,l) in pool]})

    def run(self):
        try:
            if self.iteration==0:
                self.reproduce()
            while self.iteration<self.args.attempts and self.best>=111.:
                self.check()
                proposal=self.proposal()
                self.iteration+=1
                if proposal is None:
                    self.stats["no_topology"]+=1
                    continue
                path,planned,_=proposal
                if path in self.attempted:
                    continue
                self.attempted.add(path)
                self.stats["global_order_proposals"]+=1
                trial={"iteration":self.iteration,"path":path,"family":self.family(path),
                       "changed_arcs":len(set(zip(path,path[1:]))-self.original_edges),
                       "shared_initial_visits":next((j for j,(a,b) in enumerate(zip(path,self.reference))
                                                     if a!=b),len(path)),
                       "geometry_proposal_exceptions":planned}
                self.hist.append(trial)
                self.run_route(path,trial)
                if time.monotonic()-self.last_saved>=20.:
                    self.save()
                    self.last_saved=time.monotonic()
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print(dict(self.stats),"orders",len(self.orders),"novel",len(self.novel),
                  "best",self.best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",type=Path)
    p.add_argument("--seconds",type=float,default=105.)
    p.add_argument("--attempts",type=int,default=90)
    p.add_argument("--seed",type=int,default=101001)
    p.add_argument("--partial-feedback",action="store_true")
    p.add_argument("--reverse",action="store_true")
    p.add_argument("--learn",action="store_true")
    a=p.parse_args()
    opt=Search(a)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    opt.run()
