"""Small-only reconstruction between measured shell-crossing transfers."""
import argparse
from collections import Counter, defaultdict
from itertools import permutations
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from continuous_windows_easy import FILES, SEED, WindowSearch
from elite_easy import anchor_bank, scan_archive
from global_dp_easy import Grid, evaluator
from global_event_easy import geometry
from kttsp_esa_solver import chromosome, validate
from pareto_easy import BudgetExpired, arrival, write_json
from small_v2 import atomic_json

GOALS=(110.999,108.,105.,101.,112.)
GROUPS=((4,11,17),(16,27,32),(18,23,34))


def resource(legs,limit):
    return sum(l[4]>limit for l in legs)


class Search:
    cost=WindowSearch.cost

    def __init__(self,args):
        self.args=args
        self.started=time.monotonic()
        self.deadline=self.started+args.seconds
        self.rng=random.Random(args.seed)
        self.module=evaluator()
        self.udp=self.module.udp
        self.limit=self.udp.dv_threshold
        self.extra=self.udp.dv_threshold_exception+1e-6
        self.normal,self.allowed=geometry(self.udp,self.module.MU_MOON)[0]
        self.group={node:i for i,g in enumerate(GROUPS) for node in g}
        raw=json.loads(SEED.read_text())
        self.base=validate(self.udp,raw["path"],raw["transfers"])
        if not self.base or not self.base["valid"]:
            raise ValueError("ESA rejected incumbent")
        self.best=self.base["fitness"][0]
        self.seed_path=tuple(raw["path"])
        self.seed_legs=tuple(map(tuple,raw["transfers"]))
        self.best_generated=None
        archive,_=scan_archive(self.udp)
        self.events=anchor_bank(archive)
        self.stats=Counter()
        self.hist=[]
        self.orders=set()
        self.finishes=set()
        self.seen=set()
        self.attempted=set()
        self.grids={}
        self.cache={}
        self.out=args.output
        self.out.mkdir(parents=True,exist_ok=True)
        self.job=uuid4().hex[:10]
        self.tails=defaultdict(list)
        for rows in json.loads(args.atlas.read_text())["windows"]:
            tail=tuple(map(tuple,rows))
            if resource(tail,self.limit)==1:
                self.tails[tail[0][0],tuple(l[1] for l in tail)].append(tail)
        self.tails[self.seed_legs[-3][0],self.seed_path[-3:]].append(self.seed_legs[-3:])
        # Complete verified paths provide diverse segment orders and measured
        # first-crossing windows. A retimed child is independently validated.
        self.parents={path:min(rows,key=lambda x:x[0])[1] for path,rows in archive.items()}
        self.parents[self.seed_path]=self.seed_legs
        self.high_paths=defaultdict(set)
        for p in (FILES/"small_v5").glob("*/*checkpoint.json"):
            for item in json.loads(p.read_text())["attempted"]:
                prefix=tuple(item[:-2])
                if len(prefix)==46:
                    group=tuple(sorted(set(range(49))-set(prefix)))
                    self.high_paths[prefix[-1],group].add(prefix)
        if args.resume:
            old=json.loads(args.resume.read_text())
            self.attempted.update(tuple(x) for x in old["attempted"])
            self.orders.update(map(tuple,old.get("orders",())))
            self.finishes.update(old.get("finish_bins",()))
            row=old.get("best_generated")
            if row:
                result=validate(self.udp,row["path"],row["transfers"])
                if result and result["valid"]:
                    self.best=min(self.best,result["fitness"][0])
                    self.best_generated=row

    def check(self):
        if time.monotonic()>=self.deadline:
            raise BudgetExpired

    def grid(self,reverse,fine=False):
        key=reverse,fine
        if key not in self.grids:
            step=.03125 if fine else .0625
            phase=self.args.phase%step
            grid=Grid(self.module,step,phase,self.deadline,finish_day=112. if reverse else None,
                      directional=True)
            grid.horizon=112. if reverse else 116.
            if reverse:
                # Grid reflects ephemeris velocities in reverse time. Its
                # directional prefilter also needs the reflected h_z sign.
                grid.z_speed = -grid.z_speed
            self.grids[key]=grid
        return self.grids[key]

    def edge_options(self,a,b,clock,reverse,allow,fine=False):
        grid=self.grid(reverse,fine)
        measured=[]
        cap=self.extra if allow else self.limit
        for l in self.events.get((a,b),()):
            if l[4]<=cap and (arrival(l)<=clock+8e-7 if reverse else l[2]>=clock-8e-7):
                measured.append(l)
        rclock=max(0.,112.-clock) if reverse else clock
        for l in grid.edges(b if reverse else a,a if reverse else b,rclock,
                            3.8 if fine else 3.,9.5,allow):
            if l is None or l[3]<self.udp.min_tof:
                continue
            if reverse:
                d=112.-arrival(l)
                if d<0. or d+l[3]>clock+8e-7:
                    continue
                dv=self.cost(a,b,d,l[3])
                if dv<=cap:
                    measured.append((a,b,d,l[3],dv))
            elif l[2]>=clock-8e-7 and l[4]<=cap:
                measured.append(l)
        best={}
        for l in measured:
            e=int(l[4]>self.limit)
            rank=-l[2] if reverse else arrival(l)
            if e not in best or rank<best[e][0]:
                best[e]=(rank,l)
        return [l for _,l in best.values()]

    def chain(self,path,clock,budget,reverse=False,fine=False):
        """Propagate resource-time labels between the two boundary orbits."""
        key=path,clock,budget,reverse,fine
        if key in self.cache:
            return self.cache[key]
        pairs=list(zip(path,path[1:]))
        if reverse:
            pairs.reverse()
        need=[0]*(len(pairs)+1)
        for i in range(len(pairs)-1,-1,-1):
            need[i]=need[i+1]+int(not self.normal[pairs[i]])
        states={0:(clock,())}
        depth=0
        for i,(a,b) in enumerate(pairs):
            self.check()
            nxt={}
            for used,(epoch,legs) in states.items():
                if used+need[i]>budget:
                    continue
                for l in self.edge_options(a,b,epoch,reverse,used+need[i+1]<budget,fine):
                    e=used+int(l[4]>self.limit)
                    t=l[2] if reverse else arrival(l)
                    old=nxt.get(e)
                    if e+need[i+1]<=budget and (old is None or (t>old[0] if reverse else t<old[0])):
                        nxt[e]=(t,(l,*legs) if reverse else (*legs,l))
            if not nxt:
                states={}
                break
            depth+=1
            states={}
            best=math.inf
            for used,(clock,legs) in sorted(nxt.items()):
                rank=-clock if reverse else clock
                if rank<best:
                    states[used]=(clock,legs)
                    best=rank
        result=states,depth
        if len(self.cache)>=4800:
            self.cache.clear()
        self.cache[key]=result
        return result

    def partitions(self,route):
        mids=[i for i in range(3,len(route)-3) if route[i] in self.group]
        if len(mids)!=3 or mids!=list(range(mids[0],mids[0]+3)):
            return
        m=mids[0]
        if m<5 or len(route)-m<6:
            return
        yield route,m
        if not self.args.rebuild:
            return
        # Exchange membership across the two high-shell segments and solve
        # the local ordering on each side. The middle low group stays intact.
        left=route[m-4:m]
        right=route[m+3:m+7]
        if len(left)!=4 or len(right)!=4:
            return
        before,after=route[m-5],route[m+7] if m+7<len(route) else None
        if after is None:
            return
        choices=[]
        for i in range(4):
            for j in range(4):
                l=list(left)
                r=list(right)
                l[i],r[j]=r[j],l[i]
                lhs=[p for p in permutations(l)
                     if all(self.allowed[a,b] for a,b in zip((before,*p),(*p,route[m])))]
                rhs=[p for p in permutations(r)
                     if all(self.allowed[a,b] for a,b in zip((route[m+2],*p),(*p,after)))]
                for x in lhs[:3]:
                    for y in rhs[:3]:
                        candidate=(*route[:m-4],*x,*route[m:m+3],*y,*route[m+7:])
                        if sum(not self.normal[a,b] for a,b in zip(candidate,candidate[1:]))<=4:
                            choices.append(candidate)
        self.rng.shuffle(choices)
        for p in choices[:self.args.variants]:
            self.stats["cross_segment_exchanges"]+=1
            yield p,m

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
        value=result["fitness"][0]
        self.stats["esa_feasible"]+=1
        self.orders.add(path)
        self.finishes.add(round(value,7))
        trial["valid_days"]=min(value,trial.get("valid_days",1e6))
        row={"path":path,"transfers":legs,"validation":result,"chromosome":chromosome(path,legs),
             "seed":self.args.seed,"experiment":dict(trial)}
        if self.best_generated is None or value<self.best_generated["validation"]["fitness"][0]:
            self.best_generated=row
        if value<self.best:
            second=validate(evaluator().udp,path,legs)
            if second and second["valid"] and second["fitness"][0]<self.best:
                self.best=second["fitness"][0]
                self.stats["records"]+=1
                write_json(self.out/f"easy_v6_{self.job}_{self.stats['records']:04}.json",row)

    def experiment(self,route,m,tail,goal,trial,fine=False):
        start_group=route[:4]
        left,middle,right=route[3:m],route[m-1:m+4],route[m+3:]
        earliest,depth=self.chain(start_group,0.,1,False,fine)
        trial["initial_depth"]=depth
        if not earliest:
            return
        end=tail[0][2]
        # Reserve the four cross-shell exceptions; the fifth may be spent
        # inside any segment or within the middle group's timed passage.
        minimum=lambda p:sum(not self.normal[a,b] for a,b in zip(p,p[1:]))
        nl,nm,nr=minimum(left),minimum(middle),minimum(right)
        reverse,rd=self.chain(right,end,4-1-nl-nm,True,fine)
        trial["right_depth"]=max(rd,trial.get("right_depth",0))
        middle_windows=[]
        if self.args.rebuild:
            for ru,(due,post) in reverse.items():
                passages,_=self.chain(middle,due,3-ru-nl,True,fine)
                for mu,(latest,bridge) in passages.items():
                    middle_windows.append((latest,ru+mu,bridge,post))
                    self.stats["measured_middle_windows"]+=1
        for used,(release,initial) in earliest.items():
            forward,ld=self.chain(left,release,4-used-nm-nr,False,fine)
            trial["left_depth"]=max(ld,trial.get("left_depth",0))
            for lu,(t,pre) in forward.items():
                for latest,spent,bridge,post in middle_windows:
                    if used+lu+spent>4:
                        continue
                    slack=latest-t
                    trial["bridge_gate_gap"]=max(slack,trial.get("bridge_gate_gap",-1e6))
                    if slack>=-8e-7:
                        self.submit((*initial,*pre,*bridge,*post,*tail),trial)
                crossing,md=self.chain(middle,t,4-used-lu-nr,False,fine)
                trial["middle_depth"]=max(md,trial.get("middle_depth",0))
                for mu,(finish,bridge) in crossing.items():
                    for ru,(deadline,post) in reverse.items():
                        if used+lu+mu+ru>4:
                            continue
                        gap=deadline-finish
                        trial["middle_gap"]=max(gap,trial.get("middle_gap",-1e6))
                        if gap>=-8e-7:
                            self.submit((*initial,*pre,*bridge,*post,*tail),trial)
                    # A forward measurement gives a directly comparable
                    # terminal-gate gap even when backward sampling stalls.
                    timed,fd=self.chain(right,finish,4-used-lu-mu,False,fine)
                    trial["forward_right_depth"]=max(fd,trial.get("forward_right_depth",0))
                    for _,(arr,post) in timed.items():
                        gap=end-arr
                        trial["terminal_gap"]=max(gap,trial.get("terminal_gap",-1e6))
                        if gap>=-8e-7:
                            self.submit((*initial,*pre,*bridge,*post,*tail),trial)

    def subset_bridge(self,route,m,tail,trial):
        """Reassign membership across the middle crossing and time the repair."""
        reach=self.args.radius
        lo=max(3,m-reach)
        hi=min(len(route)-2,m+3+reach)
        if hi-lo>self.args.max_window:
            return
        head,ending=route[:lo+1],route[hi:]
        nodes=route[lo+1:hi]
        target=ending[0]
        slots={v:1<<i for i,v in enumerate(nodes)}
        full=(1<<len(nodes))-1
        initial,hd=self.chain(head,0.,2,False)
        trial["subset_prefix_depth"]=hd
        if not initial:
            return
        fixed_need=sum(not self.normal[a,b] for a,b in zip(ending,ending[1:]))
        due=tail[0][2]
        backs,rd=self.chain(ending,due,2,True)
        trial["subset_backward_depth"]=rd
        # Future geometry bounds reserve an exit from the middle low group.
        low_mask=sum(bit for v,bit in slots.items() if v in self.group)
        states={(0,head[-1],used):(day,legs) for used,(day,legs) in initial.items()}
        terminal=[]
        for layer in range(len(nodes)+1):
            nxt={}
            self.stats["subset_labels"]+=len(states)
            for (mask,a,used),(day,legs) in states.items():
                remaining=full^mask
                next_vertices=nodes if remaining else (target,)
                for b in next_vertices:
                    bit=slots.get(b,0)
                    if (bit and not remaining&bit) or not self.allowed[a,b]:
                        continue
                    if a in self.group and remaining&low_mask and b not in self.group:
                        continue
                    left=remaining^bit
                    required=(1 if b in self.group else 2 if left&low_mask else 0)+fixed_need
                    can_extra=used+required<4
                    for l in self.edge_options(a,b,day,False,can_extra):
                        e=used+int(l[4]>self.limit)
                        if e+required>4:
                            continue
                        finish=arrival(l)
                        path=(*legs,l)
                        if not remaining:
                            terminal.append((finish,e,path))
                            continue
                        key=(mask|bit,b,e)
                        if key not in nxt or finish<nxt[key][0]:
                            nxt[key]=(finish,path)
            if not nxt:
                break
            states=dict(sorted(nxt.items(),key=lambda item:(
                        item[1][0] + 1.6*item[0][2] + .4*bool((full^item[0][0])&low_mask),
                        item[0]))[:self.args.beam])
        eligible={}
        for day,e,legs in terminal:
            if e not in eligible or day<eligible[e][0]:
                eligible[e]=(day,legs)
        for used,(day,pre) in eligible.items():
            for re,(latest,post) in backs.items():
                if used+re<=4:
                    gap=latest-day
                    trial["subset_bridge_gap"]=max(gap,trial.get("subset_bridge_gap",-1e6))
                    if gap>=-8e-7:
                        self.submit((*pre,*post,*tail),trial)
            paths,_=self.chain(ending,day,4-used,False)
            for _,(finish,post) in paths.items():
                gap=due-finish
                trial["terminal_gap"]=max(gap,trial.get("terminal_gap",-1e6))
                if gap>=-8e-7:
                    self.submit((*pre,*post,*tail),trial)

    def save(self):
        atomic_json(self.out/f"v6_{self.job}_checkpoint.json",
            {"best_days":self.best,"best_generated":self.best_generated,
             "elapsed":time.monotonic()-self.started,"counts":dict(self.stats),
             "seed":self.args.seed,"rebuild":self.args.rebuild,"history":self.hist,
             "orders":sorted(self.orders),"finish_bins":sorted(self.finishes),
             "attempted":sorted(self.attempted)})

    def run(self):
        bank=defaultdict(set)
        for path in self.parents:
            bank[path[-4],tuple(sorted(path[-3:]))].add(path[:-3])
        if self.args.rebuild:
            for key,value in self.high_paths.items():
                bank[key].update(value)
        scenarios=defaultdict(list)
        ref=set(zip(self.seed_path,self.seed_path[1:]))
        for (source,ending),tails in self.tails.items():
            routes=bank[source,tuple(sorted(ending))]
            if not routes:
                continue
            ranked=sorted(routes,key=lambda p:(sum((a,b) not in ref for a,b in zip(p,p[1:])),
                                                self.rng.random()))
            if self.args.rebuild and len(ranked)>18:
                ranked=ranked[:12]+self.rng.sample(ranked[12:],min(15,len(ranked)-12))
            for g in GOALS:
                eligible=[t for t in tails if arrival(t[-1])<=g]
                if eligible:
                    tail=max(eligible,key=lambda l:(l[0][2],-arrival(l[-1])))
                    for order in ranked:
                        scenarios[g].append((order,tail))
        try:
            for goal in GOALS:
                scenarios[goal].sort(key=lambda row:(-row[1][0][2],self.rng.random()))
            steps=max(map(len,scenarios.values()),default=0)
            last=self.started
            for idx in range(steps):
                for goal in GOALS:
                    self.check()
                    if idx>=len(scenarios[goal]):
                        continue
                    order,tail=scenarios[goal][idx]
                    for route,m in self.partitions(order):
                        key=(*route,goal,round(tail[0][2],6),tail[-1][1])
                        if key in self.attempted:
                            continue
                        self.attempted.add(key)
                        trial={"goal":goal,"gate":tail[0][2],"finish":arrival(tail[-1]),
                               "crossing":tail[0][:2],"left_size":m-3,"right_size":len(route)-m-3,
                               "changed_prefix_arcs":sum(pair not in ref for pair in zip(route,route[1:]))}
                        self.hist.append(trial)
                        if self.args.subset:
                            self.subset_bridge(route,m,tail,trial)
                        else:
                            self.experiment(route,m,tail,goal,trial)
                        if not self.args.subset and self.args.rebuild and -1.5<trial.get("terminal_gap",-1e6)<0.:
                            self.stats["fine_retries"]+=1
                            self.experiment(route,m,tail,goal,trial,True)
                    if time.monotonic()-last>20:
                        self.save()
                        last=time.monotonic()
                if self.best<=101.:
                    break
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print(dict(self.stats),"attempts",len(self.hist),"best",self.best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas",type=Path,default=FILES/"small_v4/diagnostic/window_e3e82fc984_checkpoint.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",type=Path)
    p.add_argument("--seconds",type=float,default=85.)
    p.add_argument("--seed",type=int,default=86101)
    p.add_argument("--phase",type=float,default=.013)
    p.add_argument("--rebuild",action="store_true")
    p.add_argument("--subset",action="store_true")
    p.add_argument("--max-window",type=int,default=17)
    p.add_argument("--radius",type=int,default=4)
    p.add_argument("--beam",type=int,default=48)
    p.add_argument("--variants",type=int,default=5)
    args=p.parse_args()
    search=Search(args)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    search.run()
