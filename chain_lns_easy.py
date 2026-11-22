"""Annealed oriented-chain destroy/repair and coupled ESA timing."""
import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import heapq
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import minimize

from elite_easy import anchor_bank, edges, flight_time, scan_archive
from coupled_easy import restore_suffix
from focus_easy import refine_joint, retime
from global_dp_easy import Grid, evaluator
from global_event_easy import FILES, SEED, fit_transfer_window, geometry
from kttsp_esa_solver import validate
from pareto_easy import BudgetExpired, Transfers, arrival, write_json


def assign_cuts(udp,legs,rng,count,operator):
    """Bias cuts toward resource allocation, long flights, and missed epochs."""
    n=len(legs)
    required=[i for i,l in enumerate(legs) if l[4]>udp.dv_threshold]
    if operator=="reorder":
        chosen=set(required)
    elif operator=="excursion":
        chosen=set(rng.sample(required,min(3,len(required))))
    else:
        chosen=set(rng.sample(required,min(2,len(required))))
    waits=[l[2]-(arrival(legs[i-1]) if i else 0.) for i,l in enumerate(legs)]
    candidates=[]
    for i,l in enumerate(legs):
        if i in chosen:
            continue
        weight=(.30+.48*l[3]+1.9*max(0.,waits[i])) if operator!="sweep" else 1.
        candidates.append((rng.random()**(1/weight),i))
    for _,i in sorted(candidates,reverse=True):
        if len(chosen)>=count:
            break
        chosen.add(i)
    return [i+1 for i in sorted(chosen)]


def reconnect(path,cuts,normal,special,budget,cost,rng,limit,deadline,fixed_start=None):
    """Enumerate oriented fragments with a static exception-budget DP."""
    bounds=(0,*cuts,len(path))
    blocks=[path[a:b] for a,b in zip(bounds,bounds[1:])]
    directions=[]
    internal=[]
    score=[]
    for block in blocks:
        variants=[block] if len(block)==1 else [block,block[::-1]]
        directions.append(variants)
        expense=0
        air=0.
        for a,b in zip(block,block[1:]):
            if not special[a,b]:
                return [],0
            expense+=int(not normal[a,b])
        for variant in variants:
            air=sum(cost(a,b,int(not normal[a,b])) for a,b in zip(variant,variant[1:]))
            score.append((variant,air))
        internal.append(expense)
    penalty=sum(internal)
    if penalty>budget:
        return [],0
    remaining_budget=budget-penalty
    complete=(1<<len(blocks))-1
    orientation_cost=dict(score)
    moves={}
    for i,variants in enumerate(directions):
        for j,sub in enumerate(variants):
            options=[]
            a=sub[-1]
            for k,frontier in enumerate(directions):
                if i==k:
                    continue
                for z,tail in enumerate(frontier):
                    b=tail[0]
                    if special[a,b]:
                        extra=int(not normal[a,b])
                        options.append((cost(a,b,extra)+orientation_cost[tail]+rng.random()*.35,
                                        k,z,extra))
            moves[i,j]=sorted(options)

    @lru_cache(maxsize=None)
    def required(mask,i,j):
        if mask==complete:
            return 0
        answer=remaining_budget+1
        for _,k,z,extra in moves[i,j]:
            bit=1<<k
            if not mask&bit:
                answer=min(answer,extra+required(mask|bit,k,z))
                if not answer:
                    break
        return answer

    expansions=0
    heap=[]
    def walk(mask,i,j,spent,value,sequence):
        nonlocal expansions
        expansions+=1
        if expansions%256==0 and time.monotonic()>=deadline:
            raise BudgetExpired
        if expansions>=32_000:
            return
        if mask==complete:
            if sequence!=path:
                item=(-value,sequence)
                if len(heap)<limit:
                    heapq.heappush(heap,item)
                elif item>heap[0]:
                    heapq.heapreplace(heap,item)
            return
        for add,k,z,extra in moves[i,j]:
            bit=1<<k
            if mask&bit:
                continue
            use=spent+extra
            if use+required(mask|bit,k,z)<=remaining_budget:
                walk(mask|bit,k,z,use,value+add,sequence+directions[k][z])

    starts=[(orientation_cost[sub]+rng.random(),i,j) for i,variants in enumerate(directions)
            for j,sub in enumerate(variants) if fixed_start is None or sub[0]==fixed_start]
    for value,i,j in sorted(starts):
        if required(1<<i,i,j)<=remaining_budget:
            walk(1<<i,i,j,0,value,directions[i][j])
    return [item[1] for item in sorted(heap,reverse=True)],expansions


class LNS:
    def __init__(self,args):
        self.args=args
        self.started=time.monotonic()
        self.deadline=self.started+args.seconds
        self.rng=random.Random(args.seed)
        self.module=evaluator()
        self.udp=self.module.udp
        self.allowed,self.geometry_floor=geometry(self.udp,self.module.MU_MOON)
        self.archive,self.loaded=scan_archive(self.udp)
        raw=json.loads(args.incumbent.read_text())
        self.baseline=validate(self.udp,raw["path"],raw["transfers"])
        if not self.baseline or not self.baseline["valid"]:
            raise ValueError("ESA did not validate the incumbent")
        self.best_days=self.baseline["fitness"][0]
        self.best_order=tuple(raw["path"])
        self.best=tuple((*l[:4],float(self.udp.compute_transfer(*l[:4]))) for l in raw["transfers"])
        self.original_order=self.best_order
        self.events=anchor_bank(self.archive)
        if args.mode=="frontier":
            for p in sorted(FILES.glob("easy_events*summary.json")):
                events=json.loads(p.read_text()).get("priced_events",())
                for a,b,d,t,_ in events:
                    value=float(self.udp.compute_transfer(a,b,d,t))
                    if value<=self.udp.dv_threshold_exception+1e-6:
                        previous=self.events.get((a,b),())
                        leg=(a,b,d,t,value)
                        if leg not in previous:
                            self.events[a,b]=(*previous,leg)
        members=[(p,min(items,key=lambda x:x[0])) for p,items in self.archive.items()]
        self.pool={p:(entry[0],entry[1]) for p,entry in members if entry[0]<=self.best_days+15.}
        self.old_orders=set(self.archive)
        self.valid_orders=set()
        self.checked=set()
        self.tried=Counter()
        self.grids={}
        self.costs={}
        self.failed=Counter()
        self.stats=Counter()
        self.history=[]
        self.operator_counts=Counter()
        self.operator_good=Counter()
        self.novel_time=self.started
        self.material_time=self.started
        self.run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
        self.known_starts=self.alternate_starts()
        self.accepted_clock={}
        self.phase_candidates=set()
        self.best_novel=None
        self.departure_events=set()
        self.normal_neighbors=tuple(sum(1<<b for b in range(self.udp.n_tomatoes) if self.allowed[0][a,b])
                                    for a in range(self.udp.n_tomatoes))

    def alternate_starts(self):
        found={}
        for p in FILES.glob("easy_events*summary.json"):
            info=json.loads(p.read_text()).get("best_new_order")
            if info and info["days"]<=self.best_days+11.:
                order=tuple(info["path"])
                if order not in self.pool:
                    found[order]=min(found.get(order,math.inf),info["days"])
        reverse=self.best_order[::-1]
        found.setdefault(reverse,self.best_days+9.)
        return [p for p,_ in sorted(found.items(),key=lambda row:row[1])]

    def grid(self,step,phase=0.,refine=False):
        key=step,phase,refine
        if key not in self.grids:
            self.grids[key]=Grid(self.module,step,step*phase,self.deadline,refine=refine,directional=True)
            self.grids[key].horizon=132.
        return self.grids[key]

    def remember(self,leg):
        a,b,dep,tof,dv=leg
        key=(a,b,round(dep,6),round(tof,6),dv>self.udp.dv_threshold)
        if key in self.departure_events:
            return
        self.departure_events.add(key)
        old=self.events.get((a,b),())
        if leg not in old:
            self.events[a,b]=(*old,leg)
            self.stats["new_exact_events"]+=1
        cl=(a,b,int(dv>self.udp.dv_threshold))
        self.costs[cl]=min(self.costs.get(cl,math.inf),tof)

    def price(self,a,b,extra):
        key=(a,b,extra)
        if key not in self.costs:
            flights=[l[3] for l in self.events.get((a,b),())
                     if int(l[4]>self.udp.dv_threshold)==extra]
            if flights:
                self.costs[key]=min(flights)
            else:
                reverse=[l[3] for l in self.events.get((b,a),())
                         if int(l[4]>self.udp.dv_threshold)==extra]
                estimate=.44+.018*self.geometry_floor[a,b] if not extra else .42+.002*self.geometry_floor[a,b]
                self.costs[key]=(.65+min(reverse)) if reverse else estimate+.80
        return self.costs[key]+.14*min(5,self.failed[a,b])

    def submit(self,path,legs,trial,stage):
        key=tuple(l[:4] for l in legs)
        if key in self.checked:
            return None
        self.checked.add(key)
        self.stats["esa_checks"]+=1
        checked=validate(self.udp,path,legs)
        if not checked or not checked["valid"]:
            self.stats["esa_rejected"]+=1
            return None
        days=checked["fitness"][0]
        self.stats["esa_feasible"]+=1
        if path not in self.valid_orders:
            self.valid_orders.add(path)
            self.stats["distinct_feasible_orders"]+=1
            self.novel_time=time.monotonic()
        if path not in self.old_orders and (self.best_novel is None or days<self.best_novel["days"]):
            self.best_novel={"days":days,"flight_days":flight_time(legs),"path":path}
        trial["fastest_valid_days"]=min(days,trial.get("fastest_valid_days",math.inf))
        if days<=self.best_days+12.:
            previous=self.pool.get(path)
            if not previous or days<previous[0]:
                self.pool[path]=(days,tuple(legs))
            if len(self.pool)>65:
                del self.pool[max(self.pool,key=lambda p:self.pool[p][0])]
            for leg in legs:
                self.remember(leg)
        if days<self.best_days:
            if self.best_days-days>1e-6:
                self.material_time=time.monotonic()
                self.stats["material_records"]+=1
            self.best_days,self.best_order,self.best=days,path,tuple(legs)
            self.stats["records"]+=1
            file=FILES/f"easy_chain_{self.run_id}_{self.stats['records']:04d}.json"
            write_json(file,{"path":path,"transfers":legs,"arrival_day":days,
                             "validation":checked,"flight_days":flight_time(legs),
                             "attempt":{**trial,"stage":stage,"random_seed":self.args.seed,
                                        "incumbent":str(self.args.incumbent)}})
            print(f"ESA improvement {days:.12f} | {file.name}",flush=True)
        return days,tuple(legs)

    def finish(self,path,grid,prefix,trial,stage,wait=5.,fine_grid=None,flight_caps=None):
        schedules,depth=retime(grid,path,self.events,prefix=prefix,wait=wait,flight=10.,
                               fine_grid=fine_grid,flight_caps=flight_caps,
                               observer=self.remember if self.args.learn else None)
        trial["maximum_depth"]=max(depth,trial.get("maximum_depth",0))
        if not schedules and depth<len(path)-1:
            self.failed[path[depth],path[depth+1]]+=1
        accepted=[checked for schedule in schedules
                  if (checked:=self.submit(path,schedule,trial,stage)) is not None]
        if not accepted:
            return None
        return min(accepted,key=lambda x:x[0])

    def precision(self,path,raw,trial,i):
        phase=(.5,.0,.25,.75)[i%4]
        fine=self.grid(.015625,phase,refine=i%3==1)
        finer=self.grid(.0078125,(.5,.0,.25)[i%3])
        caps={l[:2]:min(9.,l[3]+.55) for l in raw if l[4]>100 or l[3]>3.4}
        checked=self.finish(path,fine,(),trial,"route re-timed with selective fine events",6.,finer,caps)
        seed=checked[1] if checked else raw
        if arrival(seed[-1])<=self.best_days+.35:
            refine_joint(self.udp,seed,self.deadline,
                         lambda s:self.submit(path,s,trial,"whole-mission continuous refinement"),
                         0.,self.rng,False)

    def route_trial(self,working,i,trial):
        value,legs=working
        path=(legs[0][0],*(l[1] for l in legs))
        options=("reorder","excursion","sweep")
        weights=[.65+1.8*(1+self.operator_good[x])/(2+self.operator_counts[x]) for x in options]
        operator=self.rng.choices(options,weights)[0]
        self.operator_counts[operator]+=1
        count=(6,7,8,9,10)[i%5]
        cuts=assign_cuts(self.udp,legs,self.rng,count,operator)
        candidates,expanded=reconnect(path,cuts,*self.allowed,self.udp.n_dv_exception,self.price,
                                       self.rng,22,self.deadline)
        trial.update(parent_days=value,operator=operator,cuts=cuts,static_labels=expanded,
                     proposed=len(candidates))
        self.stats["fragment_states"]+=expanded
        self.stats["static_proposals"]+=len(candidates)
        tested=0
        best_accepted=None
        step,phase=((.0625,.5),(.046875,.25),(.03125,.0),(.0625,.25),(.03125,.5))[i%5]
        grid=self.grid(step,phase)
        for child in candidates:
            if self.tried[child]>=2:
                continue
            self.tried[child]+=1
            tested+=1
            changed=next(k for k,(a,b) in enumerate(zip(path,child)) if a!=b)
            prefix=legs[:max(0,changed-3)]
            result=self.finish(child,grid,prefix,trial,"oriented-chain global destroy/repair")
            if not result and prefix and trial.get("maximum_depth",0)>=25:
                result=self.finish(child,grid,(),trial,"fragment order with upstream timing released")
            if result:
                if best_accepted is None or result[0]<best_accepted[0]:
                    best_accepted=result
                if result[0]<value+.8:
                    self.operator_good[operator]+=1
                if result[0]<=self.best_days+1.5:
                    self.precision(child,result[1],trial,i)
            if tested>=self.args.children:
                break
        trial["timed_children"]=tested
        self.stats["route_repairs"]+=tested
        return best_accepted

    def pivot_trial(self,working,i,trial):
        """Drive a real short transfer into the route, retiming both sides."""
        _,legs=working
        path=(legs[0][0],*(l[1] for l in legs))
        if i%5<3:
            indices=sorted(range(8,43),key=lambda j:legs[j][3],reverse=True)
            index=indices[i%14]
        else:
            index=(29,36,39,42,45,46,47)[i%7]
        a,b,dep,tof,dv=legs[index]
        exact=self.grid(.03125,(0.,.5,.25)[i%3],refine=True)
        ready=arrival(legs[index-1])
        finish=arrival(legs[index])
        trial.update(pivot=[a,b],parent_ready=ready,parent_arrival=finish)
        if index<2:
            return None
        after_required=sum(not self.allowed[0][x,y] for x,y in zip(path[index+1:],path[index+2:]))
        extras=self.udp.n_dv_exception-after_required
        required=int(not self.allowed[0][a,b])
        prefixes,_=retime(exact,path[:index+1],self.events,wait=5.5,max_exceptions=extras-required)
        staged=tuple(legs[:index])
        if staged not in prefixes:
            prefixes.append(staged)
        best=None
        for prefix in prefixes:
            early=arrival(prefix[-1])
            used=sum(l[4]>self.udp.dv_threshold for l in prefix)
            threshold=(self.udp.dv_threshold_exception if used<extras and dv>100 else self.udp.dv_threshold)
            mark=(path,index,round(early,5),i%6)
            if mark in self.phase_candidates:
                continue
            self.phase_candidates.add(mark)
            limits=(finish-.13,finish-.50,finish-.94,finish-.025,finish+.18)
            for last in limits:
                if last<=early+self.udp.min_tof:
                    continue
                found,cost=fit_transfer_window(self.udp,a,b,early,last,threshold,self.deadline,self.rng)
                self.stats["continuous_pivot_windows"]+=1
                if math.isfinite(cost):
                    trial["lowest_sampled_dv"]=min(cost,trial.get("lowest_sampled_dv",math.inf))
                for l in sorted(set(found),key=arrival)[:2]:
                    self.remember(l)
                    repaired=self.finish(path,exact,prefix+(l,),trial,"real pivot with both sides re-timed",5.)
                    if repaired and (best is None or repaired[0]<best[0]):
                        best=repaired
            if best and best[0]<=self.best_days+.4:
                self.precision(path,best[1],trial,i)
        return best

    def deform_clock(self,working,i,trial):
        """Continue through feasible clocks using a milestone objective."""
        elapsed,schedule=working
        path=(schedule[0][0],*(l[1] for l in schedule))
        j=(16,25,35,38,43,29,41,8)[i%8]
        marker=(path,j,i//8%4,round(arrival(schedule[j]),5))
        if marker in self.phase_candidates:
            return None
        self.phase_candidates.add(marker)
        trial["milestone"]=path[j+1]
        trial["initial_milestone_day"]=arrival(schedule[j])
        pool=[]

        def receive(legs):
            checked=self.submit(path,legs,trial,"feasible clock homotopy")
            if checked:
                pool.append(checked)

        end_allow=(.17,.6,1.3)[(i//8)%3]
        raw=schedule
        for strength in (.45,1.5,4.5):
            refine_joint(self.udp,raw,self.deadline,receive,0.,self.rng,False,
                         clock_objectives=((j,strength),),finish_slack=end_allow,
                         coherent_shift=(max(0,j-4),min(len(schedule),j+4),0.))
            if pool:
                chosen=min(pool,key=lambda x:arrival(x[1][j]))
                if arrival(chosen[1][j]) < arrival(raw[j])-1e-7:
                    raw=chosen[1]
        if not pool:
            return None
        shaped=min(pool,key=lambda x:arrival(x[1][j]))
        trial["new_milestone_day"]=arrival(shaped[1][j])
        for l in shaped[1]:
            self.remember(l)
        fine=self.grid(.015625,(.0,.5,.25)[i%3],refine=True)
        repaired=self.finish(path,fine,shaped[1][:j+1],trial,
                             "re-timed suffix after feasible clock deformation",6.)
        if repaired and repaired[0]<shaped[0]:
            shaped=repaired
        return shaped

    def backward_join(self,working,i,trial):
        """Search open-end suffix orders from a mission deadline backwards."""
        original,legs=working
        path=(legs[0][0],*(l[1] for l in legs))
        cut=(29,34,39,31,36,41,33,37,38)[i%9]
        tail=path[cut:]
        prefix=legs[:cut]
        mandatory_before=sum(not self.allowed[0][a,b] for a,b in zip(path[:cut],path[1:cut+1]))
        focus=assign_cuts(self.udp,legs[cut:],self.rng,min(6,len(tail)-2),"reorder")
        orders,expanded=reconnect(tail,focus,*self.allowed,self.udp.n_dv_exception-mandatory_before,
                                   self.price,self.rng,12,self.deadline,fixed_start=tail[0])
        if tail not in orders:
            orders.insert(0,tail)
        cap=self.best_days-(.08,.3,.79,1.15,.04,.52)[i%6]
        if i%7==6:
            cap=self.best_days+.30
        trial.update(cut=cut,join_tomato=path[cut],deadline_day=cap,proposed=len(orders),
                     alternate_parent_days=original)
        self.stats["backward_fragment_states"]+=expanded
        reverse=Grid(self.module,.03125,.015625 if i%2 else 0.,self.deadline,
                     finish_day=cap,directional=True)
        reverse_events={}
        for (a,b),events in self.events.items():
            reverse_events[b,a]=tuple((b,a,cap-arrival(l),l[3],l[4]) for l in events if arrival(l)<=cap)
        forward=self.grid(.015625,(.5,0.,.25)[i%3],refine=i%4==1)
        best=None
        for order in orders[:self.args.children+1]:
            key=(order,i%6,cut,round(cap,7))
            if key in self.phase_candidates:
                continue
            self.phase_candidates.add(key)
            child=path[:cut]+order
            late,depth=retime(reverse,order[::-1],reverse_events,wait=5.5,
                              max_exceptions=self.udp.n_dv_exception-mandatory_before)
            trial["best_backward_depth"]=max(depth,trial.get("best_backward_depth",0))
            for backwards in late:
                suffix=restore_suffix(self.udp,backwards,cap)
                if any(l[4]>self.udp.dv_threshold_exception+1e-6 for l in suffix):
                    continue
                for l in suffix:
                    self.remember(l)
                limit=self.udp.n_dv_exception-sum(l[4]>self.udp.dv_threshold for l in suffix)
                early,_=retime(forward,child[:cut+1],self.events,max_exceptions=limit,wait=5.5)
                for head in early:
                    gap=arrival(head[-1])-suffix[0][2]
                    trial["smallest_join_gap_days"]=min(gap,trial.get("smallest_join_gap_days",math.inf))
                    if gap<=1e-6:
                        checked=self.submit(child,head+suffix,trial,"deadline-directed chain join")
                        if checked and (best is None or checked[0]<best[0]):
                            best=checked
                    if gap<=.21:
                        refine_joint(self.udp,head+suffix,self.deadline,
                                     lambda s:self.submit(child,s,trial,"continuous join feasibility"),
                                     0.,self.rng,False,
                                     coherent_shift=(max(0,cut-3),min(len(legs),cut+5),0.))
            self.stats["backward_fragment_orders"]+=1
            # Keep exploring feasible elapsed solutions even when a
            # particular hard deadline was beyond the sampled frontier.
            if order!=tail:
                result=self.finish(child,forward,prefix,trial,"forward timing of backwards-tested order")
                if result and (best is None or result[0]<best[0]):
                    best=result
        return best

    def global_frontier(self,working,i,trial):
        """Construct routes from open endpoints using geometry and real phase events."""
        n=self.udp.n_tomatoes
        all_nodes=(1<<n)-1
        budget=self.udp.n_dv_exception
        step,phase=((.0625,.5),(.0625,0.),(.046875,.25),(.0625,.25))[i%4]
        grid=self.grid(step,phase)
        near=(.9,1.5,2.2,1.1)[i%4]
        raw_events={}
        for (a,b),legs in self.events.items():
            for special in (0,1):
                group=sorted((l for l in legs if int(l[4]>100)==special),key=lambda l:l[2])
                if not group:
                    continue
                dep=tuple(l[2] for l in group)
                minima=[None]*len(group)
                best=group[-1]
                for j in range(len(group)-1,-1,-1):
                    if arrival(group[j])<arrival(best):
                        best=group[j]
                    minima[j]=best
                raw_events[a,b,special]=dep,minima
        def from_archive(a,b,release,extra):
            data=raw_events.get((a,b,extra))
            if data is None:
                return None
            deps,legs=data
            chosen=bisect_left(deps,release-1e-6)
            return legs[chosen] if chosen<len(legs) else None

        @lru_cache(maxsize=175_000)
        def components(mask):
            count=0
            while mask:
                fringe=mask&-mask
                mask^=fringe
                count+=1
                while fringe:
                    bit=fringe&-fringe
                    fringe^=bit
                    reached=self.normal_neighbors[bit.bit_length()-1]&mask
                    mask^=reached
                    fringe|=reached
            return count-1

        @lru_cache(maxsize=50_000)
        def remainder(mask,last):
            total=0.
            remaining=mask
            maximum=0.
            while remaining:
                bit=remaining&-remaining
                a=bit.bit_length()-1
                remaining^=bit
                choices=self.normal_neighbors[a]&mask
                values=[]
                while choices:
                    b=choices&-choices
                    choices^=b
                    values.append(self.price(a,b.bit_length()-1,0))
                score=min(values,default=1.)
                total+=score
                maximum=max(maximum,score)
            return max(0.,total-maximum)

        starts=set(j for j in range(n) if components(all_nodes)<=budget)
        states={(1<<a,a,0):(0.,(a,),()) for a in starts}
        templates=sorted(self.pool.values(),key=lambda row:row[0])
        reference=working[1]
        tracks=[reference]+[row[1] for row in templates[:4] if row[1]!=reference]
        protected=[]
        for schedule in tracks:
            mask=1<<schedule[0][0]
            used=0
            labels=[]
            order=(schedule[0][0],)
            for idx,leg in enumerate(schedule):
                b=leg[1]; mask|=1<<b; order+= (b,)
                used+=leg[4]>100
                labels.append(((mask,b,used),(arrival(leg),order,schedule[:idx+1])))
            protected.append(labels)
        labels_seen=0
        for depth in range(1,n):
            grid.check()
            candidates={}
            for (mask,a,used),(day,prefix,schedule) in states.items():
                labels_seen+=1
                remaining=all_nodes^mask
                normal=[]
                special=[]
                todo=remaining
                while todo:
                    node=todo&-todo
                    todo^=node
                    b=node.bit_length()-1
                    if not self.allowed[1][a,b]:
                        continue
                    need=components(remaining)
                    if used+need>budget:
                        continue
                    forced=int(not self.allowed[0][a,b])
                    if used+need+forced>budget:
                        continue
                    rank=self.price(a,b,forced)
                    (special if forced else normal).append((rank,b,node))
                forks=sorted(normal)+sorted(special)[:(2+i%3)]
                for _,b,node in forks:
                    need=components(remaining)
                    can_extra=used+need<budget
                    options=[from_archive(a,b,day,0)]
                    if can_extra:
                        options.append(from_archive(a,b,day,1))
                    options=[l for l in options if l is not None and l[2]<=day+4.]
                    if not options or min(arrival(l) for l in options)>day+self.price(
                            a,b,int(not self.allowed[0][a,b]))+near:
                        fresh=grid.edges(a,b,day,3.,8.5,can_extra)
                        options.extend(l for l in fresh if l is not None)
                    for leg in options:
                        count=used+int(leg[4]>100)
                        end=arrival(leg)
                        if count+need>budget or end>grid.horizon:
                            continue
                        key=(mask|node,b,count)
                        if key not in candidates or end<candidates[key][0]:
                            candidates[key]=(end,prefix+(b,),schedule+(leg,))
            # Feasible archives are diverse rescue states. Outside these
            # protected labels every branch can start and visit arbitrarily.
            if i%3!=2:
                for labels in protected:
                    key,label=labels[depth-1]
                    if key not in candidates or label[0]<candidates[key][0]:
                        candidates[key]=label
            if not candidates:
                trial["reached_visits"]=depth
                return None
            def rank(item):
                (mask,last,count),(day,_,_)=item
                return day+.87*remainder(all_nodes^mask,last)+.22*count
            states=dict(sorted(candidates.items(),key=rank)[:self.args.beam])
            trial["reached_visits"]=depth+1
            trial["constructive_labels"]=labels_seen
        self.stats["constructive_labels"]+=labels_seen
        accepted=[]
        for _,path,schedule in sorted(states.values())[:12]:
            checked=self.submit(path,schedule,trial,"unrestricted phase-constrained frontier construction")
            if checked:
                accepted.append((path,checked))
        for path,checked in accepted[:3]:
            if checked[0]<=self.best_days+1.2:
                self.precision(path,checked[1],trial,i)
        return min((entry for _,entry in accepted),key=lambda row:row[0],default=None)

    def shifted_event(self,leg,offset,margin):
        a,b,dep,tof,dv=leg
        guess=dep+offset
        if guess<0. or guess+tof>131.0:
            return None
        threshold=self.udp.dv_threshold_exception+7e-7 if dv>100 else self.udp.dv_threshold-2e-7
        bounds=((max(0.,guess-margin),guess+margin),
                (max(self.udp.min_tof,tof-.23),tof+.23))

        def cost(x):
            if time.monotonic()>=self.deadline:
                raise BudgetExpired
            return float(self.udp.compute_transfer(a,b,float(x[0]),float(x[1])))

        initial=np.array([guess,tof])
        optimized=minimize(lambda x:float(x[0]+x[1]),initial,method="SLSQP",bounds=bounds,
                           constraints=({"type":"ineq","fun":lambda x:threshold-cost(x)},),
                           options={"maxiter":27,"ftol":1e-8,"eps":2e-7})
        actual=[]
        for x in (optimized.x,initial):
            dv=cost(x)
            if dv<=threshold:
                actual.append((a,b,float(x[0]),float(x[1]),dv))
        return min(actual,key=arrival,default=None)

    def resonance(self,working,i,trial):
        """Translate continuous branch seeds by approximate half-orbit cycles."""
        prior,schedule=working
        path=(schedule[0][0],*(l[1] for l in schedule))
        periods=2*np.pi*np.sqrt(self.udp.tomato_orbital_parameters[:,0]**3/self.module.MU_MOON)/86400.
        shift=(-1,-2,+1,-3,+2,-1)[i%6]
        trial["half_period_multiplier"]=shift
        trial["parent_days"]=prior
        offset_number=0
        for j,l in enumerate(schedule):
            a,b,dep,tof,_=l
            same_shell=max(periods[a],periods[b])<2*min(periods[a],periods[b])
            if same_shell:
                delta=shift*(periods[a]+periods[b])/4
                options=(delta,)
            else:
                delta=shift*max(periods[a],periods[b])/2
                fraction=min(periods[a],periods[b])
                options=(delta,delta+fraction*.5,delta-fraction*.5)
            if i%4==2 and j%2:
                continue
            for delta in options:
                marker=(l[:2],round(dep+delta,5),round(tof,5),i%3)
                if marker in self.phase_candidates:
                    continue
                self.phase_candidates.add(marker)
                self.stats["phase_translations"]+=1
                repaired=self.shifted_event(l,delta,(.075,.16,.105)[i%3])
                if repaired is not None:
                    self.remember(repaired)
                    self.stats["phase_translated_events"]+=1
                    offset_number+=1
        trial["phase_events"]=offset_number
        grid=self.grid((.03125,.015625,.03125)[i%3],(.5,0.,.25)[i%3],refine=i%3==1)
        result=self.finish(path,grid,(),trial,"whole-route timing with translated Lambert branches",6.)
        if result and result[0]<=self.best_days+.5:
            self.precision(path,result[1],trial,i)
        if i%3==2:
            return self.route_trial(result or working,i,trial) or result
        return result

    def run(self):
        diverse=sorted(self.pool.items(),key=lambda row:row[1][0])
        working=diverse[min(self.args.parent,len(diverse)-1)][1]
        stop="time budget"
        print(f"{self.args.mode} | {self.run_id} | {self.loaded} accepted archive files, "
              f"{len(diverse)} starting orders | ESA baseline {self.best_days:.12f}",flush=True)
        try:
            if self.args.mode not in ("pivots","clocks","join","frontier","resonance"):
                for j,p in enumerate(self.known_starts[:3]):
                    grid=self.grid(.0625,.5)
                    grid.check()
                    self.finish(p,grid,(),{"start":j},"structurally distinct verified restart")
            stale=0
            for i in range(self.args.iterations):
                self.grid(.0625).check()
                if i%9==0 and i:
                    pool=sorted(self.pool.values(),key=lambda x:x[0])
                    eligible=[row for row in pool if row[0]<=self.best_days+(8. if self.args.mode!="pivots" else 3.)]
                    working=self.rng.choice(eligible[:12])
                trial={"iteration":i,"mode":self.args.mode}
                self.history.append(trial)
                previous=self.stats["esa_feasible"]
                if self.args.mode=="resonance":
                    result=self.resonance(working,i,trial)
                elif self.args.mode=="frontier":
                    result=self.global_frontier(working,i,trial)
                elif self.args.mode=="join":
                    result=self.backward_join(working,i,trial)
                elif self.args.mode=="clocks" or (self.args.mode=="mixed" and i%8==6):
                    result=self.deform_clock(working,i,trial)
                elif self.args.mode=="pivots" or (self.args.mode=="mixed" and i%8==5):
                    result=self.pivot_trial(working,i,trial)
                else:
                    result=self.route_trial(working,i,trial)
                stale=stale+1 if self.stats["esa_feasible"]==previous else 0
                if result:
                    delta=result[0]-working[0]
                    temp=self.args.temperature*.92**(i%19)
                    if delta<=0 or (result[0]<=self.best_days+10. and
                                    self.rng.random()<math.exp(-delta/max(temp,.10))):
                        self.stats["accepted_annealing_moves"]+=1
                        if delta>1e-6:
                            self.stats["accepted_worse_moves"]+=1
                        working=result
                if i%4==0:
                    print(f"{dict(self.stats)} | best {self.best_days:.12f} | "
                          f"{time.monotonic()-self.started:.1f}s",flush=True)
                if stale>=self.args.stale:
                    stop="no newly feasible schedules"
                    break
                if i>=12 and time.monotonic()-self.material_time>self.args.stall:
                    stop="material-improvement plateau"
                    break
            else:
                stop="finite neighborhood sweep complete"
        except BudgetExpired:
            pass
        finally:
            file=FILES/f"easy_chain_{self.run_id}_summary.json"
            write_json(file,{"baseline":self.baseline,"seed_file":str(self.args.incumbent),
                             "seed":self.args.seed,"mode":self.args.mode,"stop":stop,
                             "best_official_days":self.best_days,"best_novel_order":self.best_novel,
                             "counts":dict(self.stats),"attempts":self.history,
                             "elapsed_seconds":time.monotonic()-self.started})
            print(f"{stop} | {dict(self.stats)} | best {self.best_days:.12f} | {file.name}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("chains","mixed","pivots","clocks","join","frontier","resonance"))
    p.add_argument("--incumbent",type=Path,default=SEED)
    p.add_argument("--seconds",type=float,default=400.)
    p.add_argument("--iterations",type=int,default=95)
    p.add_argument("--seed",type=int,default=713)
    p.add_argument("--parent",type=int,default=0)
    p.add_argument("--children",type=int,default=4)
    p.add_argument("--temperature",type=float,default=2.)
    p.add_argument("--stale",type=int,default=12)
    p.add_argument("--stall",type=float,default=360.)
    p.add_argument("--learn",action="store_true")
    p.add_argument("--beam",type=int,default=54)
    args=p.parse_args()
    if not math.isfinite(args.seconds) or not 0.<args.seconds<=1800.:
        p.error("--seconds must be in (0, 1800]")
    LNS(args).run()


if __name__=="__main__":
    main()
