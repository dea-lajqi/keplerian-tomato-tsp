"""Validated elite crossover and time-dependent subset DP for ESA Easy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from itertools import permutations, product
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

from coupled_easy import middle_orders, order_of, timed_excursion
from focus_easy import mandatory, neighborhood, refine_joint, retime
from global_dp_easy import Grid, evaluator
from kttsp_esa_solver import ROOT, validate
from pareto_easy import BudgetExpired, arrival, write_json

FILES = ROOT/"results/esa"
SEED = FILES/"easy_coupled_20261121T132814Z_f5743ddc_0001.json"


def edges(path):
    return set(zip(path,path[1:]))


def flight_time(legs):
    return sum(leg[3] for leg in legs)


def distance(a,b):
    return len(a)-len(a & b)


def signature(udp, path):
    shell = udp.tomato_orbital_parameters[:,0]>1e7
    return tuple((a,b) for a,b in zip(path,path[1:]) if shell[a]!=shell[b])


def scan_archive(udp):
    routes = {}
    accepted = 0
    for p in sorted(FILES.glob("**/easy*.json")):
        d = json.loads(p.read_text())
        if "path" not in d or "transfers" not in d:
            continue
        result = validate(udp,d["path"],d["transfers"])
        if not result or not result["valid"]:
            continue
        accepted += 1
        path = tuple(d["path"])
        legs = tuple((*leg[:4],float(udp.compute_transfer(*leg[:4]))) for leg in d["transfers"])
        routes.setdefault(path,[]).append((result["fitness"][0],legs,p.name))
    return routes, accepted


def pick_elites(udp, archive, capacity, tolerance):
    candidates = [(p,min(items,key=lambda x:x[0])) for p,items in archive.items()]
    candidates.sort(key=lambda x:x[1][0])
    floor = candidates[0][1][0]
    selected = [candidates[0]]
    candidates = [entry for entry in candidates[1:] if entry[1][0] <= floor+tolerance]
    edge_sets = {p:edges(p) for p,_ in (*selected,*candidates)}
    shell_sequences = {p:signature(udp,p) for p,_ in (*selected,*candidates)}
    while candidates and len(selected)<capacity:
        selected_shell = {shell_sequences[p] for p,_ in selected}

        def score(entry):
            p,(value,legs,_) = entry
            separation = min(distance(edge_sets[p],edge_sets[q]) for q,_ in selected)
            new_pattern = shell_sequences[p] not in selected_shell
            return (1.5*separation + 2.5*new_pattern - .68*(value-floor)
                    - .08*(flight_time(legs)-flight_time(selected[0][1][1])))
        best = max(candidates,key=score)
        candidates.remove(best)
        selected.append(best)
    return selected


def connected_children(a,b):
    """Swap blocks at common vertices with identical visited sets."""
    mask_a = mask_b = 0
    common = []
    for i,(x,y) in enumerate(zip(a,b)):
        mask_a |= 1 << x
        mask_b |= 1 << y
        if x==y and mask_a==mask_b:
            common.append(i)
    cuts = [-1,*common,len(a)]
    varying = [(l,r) for l,r in zip(cuts,cuts[1:]) if a[l+1:r]!=b[l+1:r]]
    out = set()
    for first,last in varying:
        out.add(a[:first+1]+b[first+1:last]+a[last:])
        out.add(b[:first+1]+a[first+1:last]+b[last:])
    for first in common:
        out.add(a[:first+1]+b[first+1:])
        out.add(b[:first+1]+a[first+1:])
    return out


def transplanted_children(a,b):
    """Match equal node sets even when their boundary vertices differ."""
    intervals = {}
    for left in range(len(b)-2):
        mask = 0
        for right in range(left,min(left+18,len(b))):
            mask |= 1 << b[right]
            if right-left>=2:
                intervals[mask] = b[left:right+1]
    for left in range(len(a)-2):
        mask = 0
        for right in range(left,min(left+18,len(a))):
            mask |= 1 << a[right]
            segment = intervals.get(mask)
            if segment is not None and segment!=a[left:right+1]:
                yield a[:left]+segment+a[right+1:]


def factorized_children(paths,seed):
    """Assemble several independent donor blocks with equal visit sets."""
    cuts=[0]
    for i in range(1,len(seed)):
        target=set(seed[:i])
        if sum(set(p[:i])==target for p in paths)>=max(2,len(paths)*3//4):
            cuts.append(i)
    cuts.append(len(seed))
    parts=[]
    for lo,hi in zip(cuts,cuts[1:]):
        support=set(seed[lo:hi])
        blocks={seed[lo:hi]}
        blocks.update(p[lo:hi] for p in paths if set(p[lo:hi])==support)
        parts.append(sorted(blocks))
    # The Easy archive is small. Guard the Cartesian product in case a much
    # larger archive is supplied in future.
    count=math.prod(map(len,parts))
    if count>20_000:
        raise ValueError("The exact block product is too large for this archive")
    return {tuple(node for block in blocks for node in block) for blocks in product(*parts)}


def anchor_bank(archive):
    anchors = {}
    for entries in archive.values():
        for _,schedule,_ in entries:
            for leg in schedule:
                a,b,dep,tof,dv = leg
                anchors.setdefault((a,b),{})[round(dep,6),round(tof,6),dv>100] = leg
    return {pair:tuple(d.values()) for pair,d in anchors.items()}


def anchored_retime(udp, path, anchors):
    states = {0:(0.,0.,())}
    budget = udp.n_dv_exception
    for a,b in zip(path,path[1:]):
        candidates = {}
        for used,(day,air,chain) in states.items():
            for leg in anchors.get((a,b),()):
                if leg[2]+1e-6<day:
                    continue
                count = used+int(leg[4]>udp.dv_threshold)
                if count>budget:
                    continue
                finish, flown = arrival(leg),air+leg[3]
                if count not in candidates or (finish,flown)<candidates[count][:2]:
                    candidates[count] = (finish,flown,chain+(leg,))
        if not candidates:
            return []
        states = candidates
    return [legs for _,_,legs in sorted(states.values())]


def parent_graph(paths):
    """A vertex records both the last tomato and the visited set."""
    graph={}
    for path in paths:
        mask=0
        last=None
        for node in path:
            graph.setdefault((mask,last),set()).add(node)
            mask |= 1<<node
            last=node
    return graph


def graph_schedule(grid,fine,paths,anchors,cap):
    graph=parent_graph(paths)
    expected=len(paths[0])
    normal=grid.udp.dv_threshold
    budget=grid.udp.n_dv_exception

    @lru_cache(maxsize=None)
    def reserve(mask,last):
        if mask.bit_count()==expected:
            return 0
        return min((int(last is not None and mandatory(grid,last,b))+reserve(mask|1<<b,b)
                    for b in graph.get((mask,last),())),default=budget+1)

    @lru_cache(maxsize=None)
    def route_count(mask,last):
        if mask.bit_count()==expected:
            return 1
        return sum(route_count(mask|1<<b,b) for b in graph.get((mask,last),()))

    states={(0,None,0):[(0.,0.,())]}
    visited=0
    for _ in range(expected):
        grid.check()
        candidates={}
        for (mask,a,used),labels in states.items():
            visited+=len(labels)
            for b in sorted(graph.get((mask,a),())):
                next_mask=mask|1<<b
                unused=reserve(next_mask,b)
                if used+unused+int(a is not None and mandatory(grid,a,b))>budget:
                    continue
                allow=used+unused<budget
                for day,air,chain in labels:
                    options=(None,)
                    if a is not None:
                        options=grid.edges(a,b,day,4.,10.,allow)
                        if allow and mandatory(grid,a,b):
                            options+=fine.edges(a,b,day,2.4,3.2,True)
                        options+=tuple(l for l in anchors.get((a,b),()) if l[2]+1e-6>=day)
                    for leg in options:
                        if leg is None and a is not None:
                            continue
                        count=used+int(leg is not None and leg[4]>normal)
                        finish=arrival(leg) if leg else day
                        if finish>cap or count+unused>budget:
                            continue
                        key=(next_mask,b,count)
                        label=(finish,air+(leg[3] if leg else 0.),chain+((leg,) if leg else ()))
                        candidates.setdefault(key,[]).append(label)
        states={}
        for key,labels in candidates.items():
            labels.sort(key=lambda x:(x[0],x[1]))
            early=labels[0]
            nearby=[x for x in labels if x[0]<=early[0]+.9]
            short=min(nearby,key=lambda x:(x[1],x[0]))
            little_wait=min(nearby,key=lambda x:(x[0]-x[1],x[0]))
            states[key]=list(dict.fromkeys((early,short,little_wait)))
        if not states:
            return [],visited,route_count(0,None)
    timed=sorted((label for group in states.values() for label in group),
                 key=lambda label:(label[0],label[1]))
    return [chain for _,_,chain in timed],visited,route_count(0,None)


class MultiLattice:
    """Combine actual feasible events from several departure and flight grids."""

    def __init__(self,*grids):
        self.grids=grids

    def __getattr__(self,name):
        return getattr(self.grids[0],name)

    def edges(self,a,b,day,wait,flight,allow_exception=True):
        options=[]
        for grid in self.grids:
            options.extend(leg for leg in grid.edges(a,b,day,wait,flight,allow_exception)
                           if leg is not None)
        return tuple(dict.fromkeys(options))


class EliteSearch:
    def __init__(self,args):
        self.args=args
        self.module=evaluator()
        self.udp=self.module.udp
        self.archive,self.loaded=scan_archive(self.udp)
        data=json.loads(args.seed_file.read_text())
        self.baseline=validate(self.udp,data["path"],data["transfers"])
        if not self.baseline or not self.baseline["valid"]:
            raise ValueError("The requested incumbent failed ESA validation")
        self.seed_order=tuple(data["path"])
        self.best=min((x for entries in self.archive.values() for x in entries),key=lambda x:x[0])
        self.best_days=self.best[0]
        self.anchors=anchor_bank(self.archive)
        self.rng=random.Random(args.random_seed)
        self.grids={}
        self.stats=Counter()
        self.history=[]
        self.seen=set()
        self.new_quantized=set()
        self.work=set()
        self.started=time.monotonic()
        self.deadline=self.started+args.seconds
        self.run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
        self.recent_novel=self.started
        self.last_material=self.started
        self.seeded={tuple(l[:4] for l in legs) for es in self.archive.values() for _,legs,_ in es}
        self.initial_elites=self.elites()
        self.structural_paths=set(self.archive)
        self.initial_paths=set(self.archive)
        self.best_new_route=None
        self.pending=[]

    def elites(self):
        return pick_elites(self.udp,self.archive,10,9.0)

    def grid(self,step,phase,refine=False):
        key=step,phase,refine
        if key not in self.grids:
            self.grids[key]=Grid(self.module,step,step*phase,self.deadline,refine=refine)
            self.grids[key].horizon=133.
        return self.grids[key]

    def submit(self,schedule,trial,stage):
        key=tuple(l[:4] for l in schedule)
        if key in self.seen or key in self.seeded:
            self.stats["deduplicated"]+=1
            return
        self.seen.add(key)
        path=order_of(schedule)
        self.stats["esa_checks"]+=1
        checked=validate(self.udp,path,schedule)
        if not checked or not checked["valid"]:
            self.stats["esa_rejected"]+=1
            return
        self.stats["esa_feasible"]+=1
        days=checked["fitness"][0]
        air=flight_time(schedule)
        wait=days-air
        trail=trial.get("best_accepted",math.inf)
        trial["best_accepted"]=min(days,trail)
        if path not in self.structural_paths:
            self.structural_paths.add(path)
            self.stats["new_orders"]+=1
        if path not in self.initial_paths and (self.best_new_route is None or days<self.best_new_route["days"]):
            self.best_new_route={"days":days,"path":path,"flight_days":air,"waiting_days":wait}
        fingerprint=tuple((a,b,round(dep,5),round(tof,5)) for a,b,dep,tof,_ in schedule)
        if fingerprint not in self.new_quantized:
            self.new_quantized.add(fingerprint)
            self.recent_novel=time.monotonic()
            self.stats["distinct_feasible"]+=1
        if days<=self.best_days+7.:
            if path not in self.archive:
                self.archive[path]=[]
            bucket=self.archive[path]
            if len(bucket)<5 or days<min(e[0] for e in bucket):
                bucket.append((days,tuple(schedule),stage))
                for leg in schedule:
                    a,b,dep,tof,_=leg
                    old=self.anchors.get((a,b),())
                    if leg not in old:
                        self.anchors[a,b]=(*old,leg)
        if days<self.best_days:
            if self.best_days-days>1e-5:
                self.last_material=time.monotonic()
            self.best_days=days
            self.best=(days,tuple(schedule),stage)
            self.stats["records"]+=1
            file=FILES/f"easy_elite_{self.run_id}_{self.stats['records']:04d}.json"
            write_json(file,{"path":path,"transfers":schedule,"arrival_day":days,"validation":checked,
                             "flight_days":air,"waiting_days":wait,
                             "attempt":{**trial,"stage":stage}})
            print(f"ESA record {days:.12f} days | flight {air:.6f} | wait {wait:.6f} | {file.name}",flush=True)

    def physically_allowed(self,grid,path):
        used=0
        for a,b in zip(path,path[1:]):
            used+=mandatory(grid,a,b)
            if used>self.udp.n_dv_exception or grid.plane_cost[a,b]>self.udp.dv_threshold_exception+1e-5:
                return False
        return True

    def crossover(self,elite,grid):
        paths=[p for p,_ in elite]
        proposals=set(paths)
        for i,a in enumerate(paths):
            for b in paths[i+1:]:
                proposals.update(connected_children(a,b))
                proposals.update(transplanted_children(a,b))
                proposals.update(transplanted_children(b,a))
        return [p for p in sorted(proposals) if self.physically_allowed(grid,p)]

    def estimate(self,path):
        air=0.
        for a,b in zip(path,path[1:]):
            choices=self.anchors.get((a,b))
            air+=min(l[3] for l in choices) if choices else 2.0
        return air

    def score(self,path):
        schedules=anchored_retime(self.udp,path,self.anchors)
        if schedules:
            days=arrival(schedules[0][-1])
            return days+.06*flight_time(schedules[0])
        return self.estimate(path)*1.10+8.

    def refine(self,schedules,trial,stage):
        for s in schedules:
            self.submit(s,trial,stage)
        if schedules and arrival(schedules[0][-1])<=self.best_days+.65:
            for weight in (0.,.018):
                refine_joint(self.udp,schedules[0],self.deadline,
                             lambda s:self.submit(s,trial,stage+" + continuous timing"),
                             0.,self.rng,False,flight_weight=weight)

    def clock_sweep(self,path,trial,i,fast=False):
        phase=(.5,0.,.25,.75)[i%4]
        step=(.03125,.046875,.03125,.0625)[i%4]
        coarse=self.grid(step,phase)
        known=anchored_retime(self.udp,path,self.anchors)
        self.refine(known,trial,"parent-event crossover")
        anchors=dict(self.anchors)
        current,_=retime(coarse,path,anchors,wait=5.5)
        self.refine(current,trial,"recombined order + full timing DP")
        if current and arrival(current[0][-1])<=self.best_days+(1.4 if fast else 2.6):
            for leg in current[0]:
                a,b=leg[:2]
                anchors[a,b]=(*anchors.get((a,b),()),leg)
            fine=self.grid(.015625,(0.,.5,.25)[i%3],refine=(i%3==2))
            timed,_=retime(fine,path,anchors,wait=6.)
            self.refine(timed,trial,"recombined order + fine timing DP")
            if timed and arrival(timed[0][-1])<=self.best_days+.8 and not fast:
                for l in timed[0]:
                    anchors[l[:2]]=(*anchors.get(l[:2],()),l)
                precise=self.grid(.0078125,(.5,.25,0.)[i%3])
                precise_schedule,_=retime(precise,path,anchors,wait=5.)
                self.refine(precise_schedule,trial,"recombined order + precise timing DP")

    def subset(self,elite,i,trial):
        paths=self.crossover(elite,self.grid(.0625,0.))
        windows=((17,31),(16,32),(18,31),(15,33),(16,34),(17,35),(14,31),
                 (17,33),(18,34),(13,31))
        left,right=windows[i%len(windows)]
        parent=elite[(i//3)%min(7,len(elite))]
        if i%4==0:
            parent=elite[0]
        path,(_,schedule,_)=parent
        grid=self.grid((.0625,.046875,.03125)[i%3],(.5,0.,.25,.75)[(i+i//8)%4])
        fine=self.grid(.015625,(.5,0.,.25)[i%3])
        key=(path,left,right,grid.step,grid.phase)
        if key in self.work:
            return
        self.work.add(key)
        # Use only parents with the same node support for this window. The
        # DP may join their partial routes at any compatible visited subset.
        support=set(path[left:right])
        donor_blocks={p[left:right] for p in paths if set(p[left:right])==support}
        donor_orders=[path[:left]+block+path[right:] for block in sorted(donor_blocks)]
        known={e:min(l[3] for l in options) for e,options in self.anchors.items()}
        static,expansions=middle_orders(grid,path,known,left,right,self.rng)
        all_orders=set(donor_orders)
        all_orders.update(static)
        all_orders=[p for p in sorted(all_orders) if self.physically_allowed(grid,p)]
        trial.update(window=[left,right],parent=parent[1][2],donor_blocks=len(donor_blocks),
                     static_orders=len(static),static_expansions=expansions,
                     grid_days=grid.step,phase_days=grid.phase)
        prefix=schedule[:left-1]
        candidate,labels,depth=timed_excursion(grid,fine,path,all_orders,left,right,prefix,self.anchors,
                                               schedule[right][2]+1.5,diversity=2)
        trial.update(subset_labels=labels,subset_depth=depth,timed_orders=len(candidate))
        self.stats["subset_labels"]+=labels
        self.stats["subset_windows"]+=1
        for order,chain in candidate:
            timed,_=retime(fine,order,self.anchors,chain,wait=5.)
            self.refine(timed,trial,"elite-supported subset DP and propagated suffix")
            local=dict(self.anchors)
            for l in chain:
                local[l[:2]]=(*local.get(l[:2],()),l)
            alternate,_=retime(fine,order,local,chain[:left-4],wait=5.5)
            self.refine(alternate,trial,"elite subset order with coupled timing")
            if alternate and arrival(alternate[0][-1])<=self.best_days+.7:
                self.clock_sweep(order,trial,i,fast=False)

    def graph(self,elite,i,trial):
        if self.args.mode=="lattice":
            specs=(((.03125,0.),(.0234375,.5)),((.03125,.5),(.01953125,.25)),
                   ((.015625,0.),(.0234375,.75)),((.015625,.5),(.01171875,.5)),
                   ((.0234375,.125),(.015625,.75)))[i%5]
            step,phase=specs[0]
            grid=MultiLattice(*(self.grid(s,p,refine=i%3==2) for s,p in specs))
            trial["cooperating_grids"]=specs
        else:
            step,phase=((.0625,.5),(.03125,0.),(.046875,.25),(.03125,.5),
                        (.015625,.25),(.0234375,.5),(.03125,.75))[i%7]
            grid=self.grid(step,phase)
        fine=self.grid(.015625,(.5,.25,0.)[i%3])
        paths=[path for path,_ in elite]
        proposals=self.crossover(elite,grid)
        paths.extend(p for p in sorted(proposals,key=self.score)[:(8+i%3*6)] if p not in paths)
        marker=(tuple(paths),step,phase)
        if marker in self.work:
            return
        self.work.add(marker)
        schedules,labels,orders=graph_schedule(grid,fine,paths,self.anchors,self.best_days+3.5)
        trial.update(grid_days=step,grid_phase_days=grid.phase,graph_orders=orders,
                     supported_orders=len(paths),graph_labels=labels)
        self.stats["joint_graph_labels"]+=labels
        self.stats["combined_orders"]=max(self.stats["combined_orders"],orders)
        self.refine(schedules,trial,"multi-parent route and timing DP")
        retimed=set()
        for schedule in schedules:
            path=order_of(schedule)
            if path not in retimed:
                retimed.add(path)
                if arrival(schedule[-1])<=self.best_days+1.3:
                    self.clock_sweep(path,trial,i,fast=i%3!=1)

    def factor(self,elite,i,trial):
        if not self.pending or i%9==0:
            paths=[path for path,_ in elite]
            grid=self.grid(.0625,0.)
            pool=set(paths)
            for seed in paths[:5]:
                pool.update(factorized_children(paths,seed))
            feasible=[path for path in pool if self.physically_allowed(grid,path)]
            self.stats["independent_block_orders"]=max(self.stats["independent_block_orders"],len(feasible))
            scheduled={}
            for path in feasible:
                timed=anchored_retime(self.udp,path,self.anchors)
                if timed:
                    scheduled[path]=arrival(timed[0][-1])
                    for s in timed:
                        self.submit(s,trial,"independent block crossover with parent epochs")
            # First try compatible timed crossovers, then optimistically
            # scored untimed ones. Alternate the two so all budget doesn't
            # go to children whose parent epochs already agree.
            primary=sorted(scheduled,key=lambda p:(scheduled[p],self.estimate(p),p))
            secondary=sorted((p for p in feasible if p not in scheduled),key=lambda p:(self.estimate(p),p))
            self.pending=[]
            for j in range(max(len(primary),len(secondary))):
                if j<len(primary):
                    self.pending.append(primary[j])
                if j<len(secondary):
                    self.pending.append(secondary[j])
        phase=i//max(1,len(self.pending))
        candidates=[p for p in self.pending if (p,phase) not in self.work]
        if not candidates:
            return
        path=candidates[0]
        self.work.add((path,phase))
        trial.update(path=path,phase_group=phase,structurally_new=path not in self.archive)
        self.stats["factor_timing_passes"]+=1
        self.clock_sweep(path,trial,i)

    def fringe(self,elite,i,trial):
        ordered=sorted(elite,key=lambda entry:entry[1][0])
        path,(_,schedule,parent)=ordered[(i//4)%min(len(elite),4)]
        lo,hi=((37,49),(0,12),(36,49),(1,14),(34,47),(39,49),(0,13),(35,48))[i%8]
        grid=self.grid((.0625,.046875,.03125)[i%3],(.5,0.,.25,.75)[(i+i//8)%4])
        fine=self.grid(.015625,(0.,.5,.25)[i%3])
        marker=(path,lo,hi,grid.step,grid.phase)
        if marker in self.work:
            return
        self.work.add(marker)
        prefix=schedule[:max(0,lo-1)]
        if prefix:
            budget=self.udp.n_dv_exception-sum(mandatory(grid,a,b) for a,b in zip(path[hi:],path[hi+1:]))
            advanced,_=retime(fine,path[:lo],self.anchors,wait=5.,max_exceptions=budget)
            if advanced and arrival(advanced[0][-1]) < arrival(prefix[-1]):
                if sum(l[4]>self.udp.dv_threshold for l in advanced[0])<=sum(
                        l[4]>self.udp.dv_threshold for l in prefix):
                    prefix=advanced[0]
        timed,depth=neighborhood(grid,path,schedule,self.anchors,lo,hi-lo,44,hi-lo,
                                 prefix_override=prefix,coupled=True)
        trial.update(window=[lo,hi],parent=parent,subset_depth=depth,
                     timed_orders=len(timed),grid_days=grid.step,phase_days=grid.phase)
        self.stats["endpoint_windows"]+=1
        for new_path,chain in timed[:5]:
            local=dict(self.anchors)
            for l in chain:
                local[l[:2]]=(*local.get(l[:2],()),l)
            candidate,_=retime(fine,new_path,local,chain,wait=5.)
            self.refine(candidate,trial,"elite endpoint subset and epoch propagation")
            if candidate and arrival(candidate[0][-1])<=self.best_days+.65:
                self.clock_sweep(new_path,trial,i,fast=True)

    def terminal(self,elite,i,trial):
        ordered=sorted(elite,key=lambda entry:entry[1][0])
        path,(_,parent,name)=ordered[(i//3)%min(len(ordered),4)]
        cut=(45,44,43,44,42)[i%5]
        prefix=parent[:cut]
        source_release=arrival(prefix[-1])
        trial.update(parent=name,cut=cut,release_day=source_release)
        high=[node for node in path[cut+1:] if self.udp.tomato_orbital_parameters[node,0]>1e7]
        low=[node for node in path[cut+1:] if node not in high]
        if len(low)!=3:
            return
        step,phase=((.00390625,0.),(.00390625,.5),(.001953125,.5),
                    (.0029296875,.25),(.00390625,.75))[i%5]
        grid=self.grid(step,phase)
        grid.horizon=min(grid.horizon,self.best_days+.20)
        trial.update(grid_days=step,phase_days=grid.phase)
        schedules=0
        for first in permutations(high):
            for last in permutations(low):
                order=path[:cut+1]+first+last
                marker=(order,step,grid.phase,tuple(l[:4] for l in prefix))
                if marker in self.work or not self.physically_allowed(grid,order):
                    continue
                self.work.add(marker)
                if source_release+sum(grid.udp.min_tof for _ in order[cut+1:])>=grid.horizon:
                    continue
                self.stats["terminal_timing_passes"]+=1
                timed,_=retime(grid,order,self.anchors,prefix,wait=.85,flight=6.)
                self.refine(timed,trial,"elite release with precise terminal DP")
                schedules+=len(timed)
        trial["candidate_schedules"]=schedules

    def run(self):
        info=[{"file":entry[2],"days":entry[0],"flight_days":flight_time(entry[1]),
               "directed_edges_changed":distance(edges(path),edges(self.seed_order)),
               "shell_crossings":signature(self.udp,path)} for path,entry in self.initial_elites]
        print(f"{self.args.mode} | {self.run_id} | {self.loaded} certified files | "
              f"{len(self.archive)} orders | {len(info)} elites | baseline {self.best_days:.12f}",flush=True)
        stop="time budget"
        try:
            if len(info)<2 or max(x["directed_edges_changed"] for x in info)<3:
                stop="insufficient route diversity"
                return
            stale_rounds=0
            for i in range(self.args.attempts):
                elite=self.elites()
                grid=self.grid(.0625,0.)
                grid.check()
                novel_before=self.stats["distinct_feasible"]
                trial={"mode":self.args.mode,"iteration":i}
                self.history.append(trial)
                if self.args.mode=="cross":
                    if i%8==0:
                        proposals=self.crossover(elite,grid)
                        self.stats["crossover_pool_size"]=max(self.stats["crossover_pool_size"],len(proposals))
                        self.pending=sorted(proposals,key=lambda p:(self.score(p),p))
                        for path in self.pending:
                            anchored=anchored_retime(self.udp,path,self.anchors)
                            for schedule in anchored:
                                self.submit(schedule,trial,"compatible parent-event assembly")
                    phase_group=i//max(1,len(self.pending))
                    eligible=[p for p in self.pending if (p,phase_group) not in self.work]
                    if not eligible:
                        stop="crossover sweep complete"
                        break
                    path=eligible[0]
                    self.work.add((path,phase_group))
                    trial.update(path=path,phase_group=phase_group,
                                 structurally_new=path not in self.archive)
                    self.stats["crossover_timing_passes"]+=1
                    self.clock_sweep(path,trial,i)
                elif self.args.mode=="subset":
                    self.subset(elite,i,trial)
                elif self.args.mode in ("graph","lattice"):
                    self.graph(elite,i,trial)
                elif self.args.mode=="factor":
                    self.factor(elite,i,trial)
                elif self.args.mode=="fringe":
                    self.fringe(elite,i,trial)
                else:
                    self.terminal(elite,i,trial)
                stale_rounds = stale_rounds+1 if self.stats["distinct_feasible"]==novel_before else 0
                if i%2==0:
                    print(f"{self.args.mode} | {dict(self.stats)} | best {self.best_days:.12f} | "
                          f"{time.monotonic()-self.started:.1f}s",flush=True)
                if stale_rounds>=self.args.stale or time.monotonic()-self.recent_novel>=self.args.stall:
                    stop="no new feasible schedules"
                    break
                if i>=7 and time.monotonic()-self.last_material>=self.args.stall*2:
                    stop="no material improvement across the remaining variants"
                    break
            else:
                stop="distinct trial sweep complete"
        except BudgetExpired:
            pass
        finally:
            file=FILES/f"easy_elite_{self.run_id}_summary.json"
            write_json(file,{"mode":self.args.mode,"seed":str(self.args.seed_file),"baseline":self.baseline,
                             "elite_pool":info,"archived_files_validated":self.loaded,"stop":stop,
                             "counts":dict(self.stats),"best_official_days":self.best_days,
                             "best_new_order":self.best_new_route,
                             "elapsed_seconds":time.monotonic()-self.started,"attempts":self.history})
            print(f"{self.args.mode} | {stop} | {dict(self.stats)} | best {self.best_days:.12f} | {file.name}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("cross","subset","graph","factor","lattice","fringe","terminal"))
    parser.add_argument("--seed-file",type=Path,default=SEED)
    parser.add_argument("--seconds",type=float,default=420.)
    parser.add_argument("--attempts",type=int,default=24)
    parser.add_argument("--stall",type=float,default=190.)
    parser.add_argument("--stale",type=int,default=5)
    parser.add_argument("--random-seed",type=int,default=322)
    args=parser.parse_args()
    if not math.isfinite(args.seconds) or not 0<args.seconds<=1500:
        parser.error("--seconds must be in (0, 1500]")
    EliteSearch(args).run()


if __name__=="__main__":
    main()
