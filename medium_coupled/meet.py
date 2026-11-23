"""Bidirectional, exception-aware reconstruction across a complete Medium endgame."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import pairwise
import json
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.checkpoint import atomic_json
from medium_coupled.endgame import WindowGrid
from medium_coupled.run import ROOT, solve, Paused
from medium_overnight.validation import official_udp, solution_payload, verify
from medium_v4.driver import unpack
from pareto_easy import BudgetExpired


class SideGrid(WindowGrid):
    def __init__(self, udp, release, due, step, phase, stop, reverse):
        super().__init__(udp,release,due,step,phase,stop)
        self.reverse=reverse
        if reverse:
            self.z_speed=-self.z_speed
            self.origin=due

    def state_at(self,node,day):
        if not self.reverse:
            return super().state_at(node,day)
        r,v=self.udp.tomatoes[node].eph(self.origin-day)
        return r,tuple(-x for x in v)


def labels(udp, grid, anchors, nodes, allowed, required, depth, budget, beam, start, stats):
    frontier=start
    reverse=grid.reverse
    for level in range(depth):
        candidates={}
        for (mask,a,e),(clock,chain) in frontier.items():
            grid.check()
            remaining=allowed&~mask
            required_left=required&~mask
            if (depth-level)==required_left.bit_count():
                remaining &= required_left
            while remaining:
                bit=remaining&-remaining
                remaining^=bit
                b=nodes[bit.bit_length()-1]
                edge=(b,a) if reverse else (a,b)
                options=list(anchors[edge])
                for leg in grid.edges(a,b,clock,grid.horizon,grid.horizon,e<budget):
                    if leg is None:
                        continue
                    tof=leg[3]
                    d=grid.origin-leg[2]-tof if reverse else grid.origin+leg[2]
                    dv=float(udp.compute_transfer(*edge,d,tof))
                    stats["exact_replays"]+=1
                    options.append((*edge,d,tof,dv))
                for leg in options:
                    d,f,dv=leg[2:]
                    left=grid.origin-d-f if reverse else d-grid.origin
                    spent=e+int(dv>udp.dv_threshold)
                    end=left+f
                    if (left<clock-1e-9 or end>grid.horizon+1e-9 or spent>budget
                            or dv>udp.dv_threshold_exception+1e-6):
                        continue
                    key=(mask|bit,b,spent)
                    if key not in candidates or end<candidates[key][0]:
                        candidates[key]=(end,(*chain,leg))
        stats["raw_labels"]+=len(candidates)
        # Allow different cover sets to survive even if the fastest phase
        # for one set has many nearly identical competing endpoints.
        occupancy=Counter()
        frontier={}
        for key,value in sorted(candidates.items(),key=lambda row:(row[1][0],row[0][2],row[0][0])):
            if occupancy[key[0]]>=8:
                continue
            occupancy[key[0]]+=1
            frontier[key]=value
            if len(frontier)>=beam:
                break
        if not frontier:
            break
    return frontier


def search(udp,sol,width,goal,step,shift,beam,stop,stats,accept,*,
           repair_near=False,reserve=0,anchor_rows=()):
    left=len(sol["dep"])-width
    path,dep,tof=(sol[k] for k in ("path","dep","tof"))
    nodes=path[left+1:]
    release=dep[left-1]+tof[left-1]
    anchors=defaultdict(list)
    spent=0
    for i,edge in enumerate(pairwise(path)):
        dv=float(udp.compute_transfer(*edge,dep[i],tof[i]))
        if i<left:spent+=int(dv>udp.dv_threshold)
        elif dep[i]+tof[i]<=goal:anchors[edge].append((*edge,dep[i],tof[i],dv))
    for leg in anchor_rows:
        if leg[2]+leg[3]<=goal and leg not in anchors[leg[:2]]:
            anchors[leg[:2]].append(leg)
    budget=udp.n_dv_exception-spent-reserve
    if budget<0:
        return
    mid=width//2
    full=(1<<width)-1
    required_f=(1<<max(0,mid-shift))-1
    required_b=full^((1<<min(width,mid+shift))-1)
    allowed_f=full^required_b
    allowed_b=full^required_f
    reverse_start={(1<<i,b,0):(0.,()) for i,b in enumerate(nodes) if allowed_b&(1<<i)}
    grids=[SideGrid(udp,release,goal,step,p,stop,rev) for p,rev in ((.013,False),(.007,True))]
    try:
        f=labels(udp,grids[0],anchors,nodes,allowed_f,required_f,mid,budget,beam,
                 {(0,path[left],0):(0.,())},stats)
        b=labels(udp,grids[1],anchors,nodes,allowed_b,required_b,width-mid-1,budget,beam,
                 reverse_start,stats)
        by_mask=defaultdict(list)
        for (mask,node,used),(clock,chain) in b.items():
            if mask.bit_count()==width-mid:
                by_mask[mask].append((node,used,clock,chain))
        pairs=[]
        near=[]
        best_slack=None
        for (mask,a,e),(clock,pre) in f.items():
            if mask.bit_count()!=mid:
                continue
            for z,k,t,post in by_mask[full^mask]:
                stats["complementary"]+=1
                r,d=release+clock,goal-t
                if e+k>budget:
                    stats["over_budget_joins"]+=1
                    continue
                slack=d-r-udp.min_tof
                best_slack=slack if best_slack is None else max(best_slack,slack)
                if slack>0:
                    pairs.append((d-r,a,z,budget-e-k,r,d,pre,post))
                elif repair_near and slack>=-.23:
                    near.append((slack,a,z,budget-e-k,r,d,pre,post))
        if best_slack is not None:
            stats["best_slack_after_min_tof"] = best_slack
        stats["open_joins"]+=len(pairs)
        seen=Counter()
        rng=random.Random(55201)
        def reconstruct(pre,bridge,post):
            suffix=(*pre,bridge,*post[::-1])
            return {"path":path[:left+1]+[x[1] for x in suffix],
                    "dep":dep[:left]+[x[2] for x in suffix],
                    "tof":tof[:left]+[x[3] for x in suffix]}

        for _,a,b,balance,r,d,pre,post in sorted(pairs,key=lambda row:-row[0]):
            if seen[a,b]>=2:
                continue
            if sum(seen.values())>=40:
                break
            seen[a,b]+=1
            cap=udp.dv_threshold_exception if balance else udp.dv_threshold
            stats["continuous_bridge_checks"]+=1
            fits,_=fit_transfer_window(udp,a,b,r,d,cap,stop,rng)
            exact=[leg for leg in anchors[a,b] if leg[2]>=r-1e-9 and
                   leg[2]+leg[3]<=d+1e-9 and leg[4]<=cap]
            for leg in (*fits,*exact):
                accept(reconstruct(pre,leg,post))
        used=set()
        for _,a,b,balance,r,d,pre,post in sorted(near,key=lambda row:-row[0]):
            if (a,b) in used:
                continue
            if len(used)>=8:
                break
            used.add((a,b))
            cap=udp.dv_threshold_exception if balance else udp.dv_threshold
            pad=max(.14, (r-d)*.5+.085)
            stats["relaxed_bridge_checks"]+=1
            fits,_=fit_transfer_window(udp,a,b,max(release,r-pad),min(goal,d+pad),cap,stop,rng)
            if not fits:
                continue
            leg=min(fits,key=lambda x:max(0.,r-x[2])+max(0.,x[2]+x[3]-d))
            trial=reconstruct(pre,leg,post)
            try:
                solve(udp,trial,left,1.2,0.,min(stop,time.monotonic()+4.),accept,stats)
                stats["near_join_polishes"]+=1
            except Paused:
                if time.monotonic()>=stop:
                    raise BudgetExpired
    finally:
        for grid in grids:
            stats.update(grid.stats)


def run(args):
    start=time.monotonic()
    stop=start+args.seconds
    udp=official_udp(ROOT)
    seed=unpack(json.loads(args.seed_record.read_text()))
    baseline=verify(udp,seed)[0]
    file=args.directory/"checkpoint.json"
    state=json.loads(file.read_text()) if file.exists() else {
        "best":baseline,"record":None,"tests":[],"valid_route_signatures":[]}
    signatures=set(state["valid_route_signatures"])
    done={tuple(x["key"]) for x in state["tests"]}

    def save():
        state["valid_route_signatures"]=sorted(signatures)
        atomic_json(file,state)

    try:
        for width,step,shift,gain in [(12,.0625,2,.005),(14,.0625,2,.005),
                                     (16,.0625,2,.03),(19,.0625,3,.05),
                                     (14,.046875,2,.015),(16,.046875,2,.025),
                                     (19,.046875,2,.055),(12,.03125,2,.005)]:
            key=(width,step,shift,gain)
            if key in done:
                continue
            if time.monotonic()>=stop:
                break
            begin=time.monotonic()
            local_stop=min(stop,begin+args.trial_seconds)
            stats=Counter()
            def accept(sol):
                stats["official_checks"]+=1
                try:
                    fit=verify(udp,sol)
                except ValueError:
                    return False
                stats["valid_schedules"]+=1
                signatures.add(hashlib.sha256(bytes(sol["path"])).hexdigest())
                if fit[0]<state["best"]:
                    independent=verify(official_udp(ROOT),sol)
                    payload=solution_payload(sol,independent,udp=udp,provenance={"plan":key})
                    path=args.directory/"records"/f"{fit[0]:.12f}_{uuid4().hex[:8]}.json"
                    atomic_json(path,{"solution":sol,**payload})
                    state.update(best=fit[0],record=str(path.resolve()))
                    save()
                    print("RECORD",fit,path,flush=True)
                return True
            try:
                goal=state["best"]-gain if args.goal is None else min(args.goal,state["best"]-gain)
                search(udp,seed,width,goal,step,shift,args.beam,
                       local_stop,stats,accept,repair_near=args.repair_near)
            except BudgetExpired:
                stats["time_cap"]=1
            state["tests"].append({"key":key,"seconds":time.monotonic()-begin,"stats":dict(stats)})
            save()
            print(state["tests"][-1],flush=True)
    finally:
        state["wall_seconds"]=state.get("wall_seconds",0.)+time.monotonic()-start
        save()
    print("FINISH",state["best"],len(signatures),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--seed-record",required=True,type=Path)
    p.add_argument("--seconds",type=float,default=115)
    p.add_argument("--trial-seconds",type=float,default=15)
    p.add_argument("--beam",type=int,default=230)
    p.add_argument("--repair-near",action="store_true")
    p.add_argument("--goal",type=float)
    run(p.parse_args())
