"""Complete-schedule population search with exact two-window order repair."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations, permutations
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from uuid import uuid4

from continuous_windows_easy import FILES, SEED
from elite_easy import anchor_bank, scan_archive
from focus_easy import refine_joint, retime
from global_dp_easy import Grid, evaluator
from global_event_easy import geometry
from kttsp_esa_solver import chromosome, validate
from pareto_easy import BudgetExpired, arrival, write_json


def atomic_json(path, row):
    tmp = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    with tmp.open("x",encoding="utf-8") as stream:
        json.dump(row,stream,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def repair_orders(path, left, right, normal, allowed, keep, rng):
    """Swap membership between disjoint holes, then solve both permutations."""
    a,b = left
    c,d = right
    first, second = tuple(path[a:b]),tuple(path[c:d])
    bridge = (*zip(path[:a-1],path[1:a]),*zip(path[b:c-1],path[b+1:c]),*zip(path[d:],path[d+1:]))
    reserved = sum(not normal[x,y] for x,y in bridge)
    if reserved>5:
        return []
    @lru_cache(maxsize=150)
    def fillings(nodes, begin, end):
        rows=[]
        for perm in permutations(nodes):
            edges=tuple(zip((begin,*perm),(*perm,end)))
            if all(allowed[x,y] for x,y in edges):
                extra=sum(not normal[x,y] for x,y in edges)
                rows.append((extra,perm))
        return rows
    choices=Counter()
    for count in (0,1,2):
        for swap1 in combinations(first,count):
            for swap2 in combinations(second,count):
                one=tuple(sorted((set(first)-set(swap1))|set(swap2)))
                two=tuple(sorted((set(second)-set(swap2))|set(swap1)))
                for e1,p1 in fillings(one,path[a-1],path[b]):
                    for e2,p2 in fillings(two,path[c-1],path[d]):
                        if e1+e2+reserved<=5 and (p1!=first or p2!=second):
                            child=(*path[:a],*p1,*path[b:c],*p2,*path[d:])
                            choices[child]=4*count-.35*(e1+e2)+rng.random()*2
    return [p for p,_ in choices.most_common(keep)]


class Search:
    def __init__(self,args):
        self.args=args
        self.start=time.monotonic()
        self.deadline=self.start+args.seconds
        self.rng=random.Random(args.seed)
        self.module=evaluator()
        self.udp=self.module.udp
        self.allowed,_=geometry(self.udp,self.module.MU_MOON)
        audit=json.loads((FILES/"small_v2/audit.json").read_text())
        self.archive,_=scan_archive(self.udp)
        self.pool={}
        for path,rows in self.archive.items():
            self.pool[path]=min(rows,key=lambda r:r[0])[:2]
        for row in audit["representatives"]:
            self.add(row)
        if args.resume:
            old=json.loads(args.resume.read_text())
            for row in old["pool"]:
                self.add(row)
            state=old["rng"]
            self.rng.setstate((state[0],tuple(state[1]),state[2]))
        for library in args.library:
            for row in json.loads(library.read_text())["pool"]:
                self.add(row)
        self.start_orders=set(self.pool)
        self.best=json.loads(SEED.read_text())
        self.best_days=validate(self.udp,self.best["path"],self.best["transfers"])["fitness"][0]
        for path,(days,legs) in self.pool.items():
            if days<self.best_days:
                self.best_days=days
                self.best={"path":path,"transfers":legs}
        self.raw_best=self.best_days
        self.anchors=anchor_bank(self.archive)
        self.stats=Counter()
        self.arm=Counter()
        self.arm_gain=Counter()
        self.routes=set()
        self.seen=set()
        self.history=[]
        self.tries=Counter()
        self.grids={}
        self.job=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
        self.output=args.output
        self.output.mkdir(parents=True,exist_ok=True)
        self.checkpoint=self.output/f"v2_{self.job}_checkpoint.json"
        self.last=time.monotonic()

    def add(self,row):
        path=tuple(row["path"])
        result=validate(self.udp,path,row["transfers"])
        if not result or not result["valid"]:
            return
        days=result["fitness"][0]
        if path not in self.pool or days<self.pool[path][0]:
            legs=tuple((*l[:4],float(self.udp.compute_transfer(*l[:4]))) for l in row["transfers"])
            self.pool[path]=(days,legs)

    def grid(self,step,phase,refine=False):
        key=(step,phase,refine)
        if key not in self.grids:
            self.grids[key]=Grid(self.module,step,phase,self.deadline,refine=refine,directional=True)
            self.grids[key].horizon=min(self.udp.max_time,self.best_days+16)
        return self.grids[key]

    def submit(self,path,legs,method):
        key=tuple(l[:4] for l in legs)
        if key in self.seen:
            return
        self.seen.add(key)
        self.stats["checks"]+=1
        result=validate(self.udp,path,legs)
        if not result or not result["valid"]:
            self.stats["rejected"]+=1
            return
        self.stats["feasible"]+=1
        days=result["fitness"][0]
        self.raw_best=min(self.raw_best,days)
        if self.stats["feasible"]==1:
            self.raw_best=days
        if path not in self.routes:
            self.routes.add(path)
            self.stats["distinct"]+=1
            if path not in self.start_orders:
                self.stats["novel"]+=1
                self.arm_gain[method]+=2.
        if path not in self.pool or days<self.pool[path][0]:
            self.pool[path]=(days,tuple(legs))
        if days<self.best_days:
            self.best_days=days
            self.best={"path":path,"transfers":legs,"validation":result,
                       "chromosome":chromosome(path,legs),"method":method,"seed":self.args.seed}
            self.stats["records"]+=1
            write_json(self.output/f"easy_v2_{self.job}_{self.stats['records']:04}.json",self.best)
            self.arm_gain[method]+=15.
            print("ESA RECORD",days,flush=True)

    def retain(self):
        ranked=sorted(self.pool.items(),key=lambda row:row[1][0])
        return ranked[:55]

    def save(self):
        atomic_json(self.checkpoint,{"stats":dict(self.stats),"arm_attempts":dict(self.arm),
             "arm_reward":dict(self.arm_gain),"best_days":self.best_days,
             "best_generated":self.raw_best if self.stats["feasible"] else None,
             "best":self.best,"orders":sorted(self.routes),"novel_orders":sorted(self.routes-self.start_orders),
             "elapsed_seconds":time.monotonic()-self.start,"seed":self.args.seed,"rng":self.rng.getstate(),
             "recent_trials":self.history[-180:],"pool":[{"path":p,"transfers":l} for p,(_,l) in self.retain()]})

    def parent(self):
        ranked=self.retain()
        shortlist=ranked[:(12 if self.stats["iterations"]%3 else 46)]
        return min(self.rng.sample(shortlist,min(6,len(shortlist))),
                   key=lambda e:self.tries[e[0]]+.12*(e[1][0]-self.best_days)+self.rng.random())

    def timed(self,route,old,left,method,trial):
        i=self.stats["iterations"]
        prefix=old[:left]
        anchors=dict(self.anchors)
        for leg in old:
            anchors[leg[:2]]=(*anchors.get(leg[:2],()),leg)
        step,phase=((.0625,.009),(.046875,.021),(.0625,.045),(.03125,.011))[i%4]
        grid=self.grid(step,phase)
        schedules,depth=retime(grid,route,anchors,prefix=prefix,wait=5.5,flight=10.)
        trial["reached"]=max(trial.get("reached",0),depth)
        for schedule in schedules:
            self.submit(route,schedule,method)
            if self.args.joint and method=="exchange" and i%3==0 and arrival(schedule[-1])<self.best_days+1.2:
                self.phase(route,schedule,method,trial)
            if arrival(schedule[-1])<=self.best_days+.2 and i%4==0:
                refine_joint(self.udp,schedule,self.deadline,lambda s:self.submit(route,s,method),
                             0.,self.rng,False)

    def exchange(self,path,legs,method,trial):
        critical=[i+1 for i,l in enumerate(legs) if l[4]>self.udp.dv_threshold and 3<i<44]
        for j in range(12):
            first=self.rng.choice(critical) if critical and j%2==0 else self.rng.randrange(2,45)
            second=self.rng.randrange(2,45)
            a,c=sorted((first,second))
            width=(4,5,6)[(self.stats["iterations"]+j)%3]
            if c-a<=width or c+width>=len(path):
                continue
            paths=repair_orders(path,(a,a+width),(c,c+width),*self.allowed,5,self.rng)
            if paths:
                trial["holes"]=[a,c,width]
                self.stats["repair_orders"]+=len(paths)
                for route in paths[:3]:
                    trial["changed_arcs"]=max(trial.get("changed_arcs",0),
                        len(set(zip(route,route[1:]))-set(zip(path,path[1:]))))
                    self.timed(route,legs,a-1,method,trial)
                return

    def phase(self,path,legs,method,trial):
        i=self.stats["iterations"]
        waits=[legs[j+1][2]-arrival(l) for j,l in enumerate(legs[:-1])]
        targets=sorted(range(len(waits)),key=lambda j:-(waits[j]+.12*legs[j][3]))
        pivot=targets[i%min(18,len(targets))]
        left=max(0,pivot-(i%9))
        # The archived leg is deliberately removed for this query. A later
        # first arrival may lead to a different suffix and exception allocation.
        a,b=path[left:left+2]
        clock=arrival(legs[left-1]) if left else 0.
        offset=(.08,.2,.42,.75,1.12)[i%5]
        grid=self.grid(.046875,(i%8)*.0053,True)
        choices=grid.edges(a,b,clock+offset,2.75,9.5,True)
        for leg in choices:
            if leg is None or leg[3]<self.udp.min_tof:
                continue
            self.stats["phase_branches"]+=1
            self.repair_phase(
                path,legs,(*legs[:left],leg),method,trial)

    def repair_phase(self,path,legs,prefix,method,trial):
        anchors=dict(self.anchors)
        for leg in legs:
            anchors[leg[:2]]=(*anchors.get(leg[:2],()),leg)
        grid=self.grid(.0625,(self.stats["iterations"]%8)*.0061)
        options,depth=retime(grid,path,anchors,prefix=prefix,wait=6.,flight=10.)
        trial["reached"]=max(depth,trial.get("reached",0))
        for candidate in options:
            self.submit(path,candidate,method)

    def run(self):
        try:
            while self.best_days>101 and time.monotonic()<self.deadline:
                self.stats["iterations"]+=1
                method=self.args.mode
                if method=="adaptive":
                    iteration=self.stats["iterations"]
                    method=max(("exchange","phase"),key=lambda k:
                        self.arm_gain[k]/max(1,self.arm[k])+1.1*math.sqrt(math.log(iteration+1)/(self.arm[k]+1)))
                self.arm[method]+=1
                path,(value,legs)=self.parent()
                self.tries[path]+=1
                trial={"method":method,"parent":value}
                self.history.append(trial)
                getattr(self,method)(path,legs,method,trial)
                if time.monotonic()-self.last>=25.:
                    self.save()
                    self.last=time.monotonic()
                    print(dict(self.stats),"best",self.best_days,flush=True)
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print(dict(self.stats),"best",self.best_days,self.checkpoint,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",type=Path)
    p.add_argument("--library",type=Path,nargs="*",default=[])
    p.add_argument("--joint",action="store_true")
    p.add_argument("--mode",choices=("exchange","phase","adaptive"),default="adaptive")
    p.add_argument("--seconds",type=float,default=180.)
    p.add_argument("--seed",type=int,default=82201)
    opt=Search(p.parse_args())
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    opt.run()
