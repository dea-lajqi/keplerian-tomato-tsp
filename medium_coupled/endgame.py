"""Time-dependent Medium endgame reconstruction with exception-aware subset labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import pairwise
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pykep as pk
from global_dp_easy import Grid
from pareto_easy import BudgetExpired
from medium_coupled.run import ROOT, BASE, ARCHIVE, solve, Paused
from medium_v4.driver import unpack
from medium_coupled.checkpoint import atomic_json
from medium_overnight.validation import official_udp, solution_payload, verify


class WindowGrid(Grid):
    def __init__(self, udp, origin, due, step, phase, stop):
        module = SimpleNamespace(udp=udp, pk=pk, MU_MOON=4.9048695e12)
        super().__init__(module, step, phase, stop, directional=True)
        self.horizon = due-origin
        self.origin = origin

    def state_at(self, node, day):
        return self.udp.tomatoes[node].eph(self.origin+day)


def decode(udp, sol, left, due, step, phase, stop, stats, result, *,
           reserve=0, endpoint=None, targets_override=None, extra_anchors=()):
    path, dep, tof = (sol[key] for key in ("path","dep","tof"))
    edges = list(pairwise(path))
    costs = [float(udp.compute_transfer(*edge,d,f)) for edge,d,f in zip(edges,dep,tof)]
    stats["anchor_evaluations"] += len(costs)
    pre_spend = sum(v>udp.dv_threshold for v in costs[:left])
    budget = udp.n_dv_exception-pre_spend-reserve
    if budget < 0:
        return
    release = dep[left-1]+tof[left-1]
    targets = path[left+1:] if targets_override is None else targets_override
    grid = WindowGrid(udp, release, due, step, phase, stop)
    cap = udp.dv_threshold_exception+1e-6
    anchors = defaultdict(list)
    for edge,d,f,v in zip(edges,dep,tof,costs):
        if d>=release-1e-9 and d+f<=due:
            anchors[edge].append((d,f,v))
    for a,b,d,f,v in extra_anchors:
        if d>=release-1e-9 and d+f<=due:
            anchors[a,b].append((d,f,v))
    frontier = {(0,path[left],0):(release,())}
    try:
        for depth in range(len(targets)):
            if time.monotonic()>=stop:
                raise BudgetExpired
            update = {}
            for (mask,a,used),(clock,chain) in sorted(frontier.items(),key=lambda row:row[1][0]):
                limit = due if depth==len(targets)-1 else due-(len(targets)-depth-1)*udp.min_tof
                for i,b in enumerate(targets):
                    if (mask & (1<<i) or limit<clock+udp.min_tof
                            or (endpoint is not None
                                and (b==endpoint)!=(depth==len(targets)-1))):
                        continue
                    samples = grid.edges(a,b,clock-release,grid.horizon,grid.horizon,used<budget)
                    options = list(anchors[a,b])
                    for leg in samples:
                        if leg is not None:
                            d, f = release+leg[2],leg[3]
                            stats["exact_replays"] += 1
                            options.append((d,f,float(udp.compute_transfer(a,b,d,f))))
                    for d,f,v in options:
                        e = used+int(v>udp.dv_threshold)
                        day = d+f
                        if d+1e-9<clock or day>limit+1e-9 or e>budget or v>cap:
                            continue
                        key = (mask|(1<<i),b,e)
                        if key not in update or day<update[key][0]:
                            update[key] = (day,(*chain,(b,d,f)))
            stats["labels"] += len(update)
            stats["max_depth"] = max(stats["max_depth"],depth+1 if update else depth)
            frontier = update
            if not frontier:
                break
        for (mask,_,_),(_,chain) in sorted(frontier.items(),key=lambda row:row[1][0]):
            if mask.bit_count()==len(targets):
                result({"path":path[:left+1]+[x[0] for x in chain],
                        "dep":dep[:left]+[x[1] for x in chain],
                        "tof":tof[:left]+[x[2] for x in chain]})
    finally:
        stats.update(grid.stats)


def run(args):
    start=time.monotonic()
    stop=start+args.seconds
    udp=official_udp(ROOT)
    original=unpack(json.loads(BASE.read_text()))
    baseline=verify(udp,original)[0]
    seeds=[original]
    if args.seed_record:
        additional=unpack(json.loads(args.seed_record.read_text()))
        seeds.insert(0,additional)
    pool=json.loads((ARCHIVE/"checkpoint.json").read_text())["pool"]
    tails={tuple(x["path"][-14:]) for x in seeds}
    for item in pool:
        candidate=unpack(item)
        key=tuple(candidate["path"][-14:])
        if key not in tails:
            seeds.append(candidate)
            tails.add(key)
    for sol in seeds:
        verify(udp,sol)
    folder=args.directory
    file=folder/"checkpoint.json"
    state=json.loads(file.read_text()) if file.exists() else {
        "best":min(verify(udp,s)[0] for s in seeds),"record":None,"completed":[],"trials":[]}
    from_best=seeds[0]
    if state["record"]:
        from_best=unpack(json.loads(Path(state["record"]).read_text()))
        verify(udp,from_best)
    modes=[(k,step,phase,off) for step,phase in [(0.046875,0.),(.03125,.009),(.02,.006)]
           for k,off in [(5,.03),(8,.10),(11,.20),(14,.35),(17,.50)]]
    completed={tuple(v) for v in state["completed"]}
    route_keys=set(state.get("valid_route_signatures",[]))
    try:
        for i in range(min(len(seeds),args.starts)):
            for width,step,phase,off in modes:
                key=(i,width,step,phase,off)
                if key in completed:continue
                source=from_best if i==0 else seeds[i]
                stats=Counter()
                seen=set()
                finalists={}
                due=(state["best"]+args.finish_slack if args.polish_limit else
                     min(state["best"]-min(off,args.minimum_gain),baseline))
                t=time.monotonic()
                def accept(candidate, *, collect=True):
                    stats["official_checks"]+=1
                    try:
                        fit=verify(udp,candidate)
                    except ValueError:
                        return False
                    stats["valid_schedules"]+=1
                    order=tuple(candidate["path"])
                    seen.add(order)
                    route_keys.add(hashlib.sha256(bytes(order)).hexdigest())
                    state["valid_route_signatures"]=sorted(route_keys)
                    if collect and (order not in finalists or fit[0]<finalists[order][0]):
                        finalists[order]=(fit[0],candidate)
                    if fit[0]<state["best"]:
                        newfit=verify(official_udp(ROOT),candidate)
                        payload=solution_payload(candidate,newfit,udp=udp,
                                       provenance={"plan":key,"continuous_polish":not collect})
                        record=folder/"records"/f"{fit[0]:.12f}_{uuid4().hex[:8]}.json"
                        atomic_json(record,{"solution":candidate,**payload})
                        state.update(best=fit[0],record=str(record.resolve()))
                        atomic_json(file,state)
                        print("RECORD",newfit,record,flush=True)
                    return True
                try:
                    decode(udp,source,180-width,due,step,phase,min(stop,t+args.trial_seconds),
                           stats,accept)
                    state["completed"].append(key)
                except BudgetExpired:
                    stats["time_cap"]=1
                    if time.monotonic()<stop:
                        state["completed"].append(key)
                if args.polish_limit and finalists:
                    for _,candidate in sorted(finalists.values(),key=lambda row:row[0])[:args.polish_limit]:
                        try:
                            solve(udp,candidate,max(0,180-width-3),2.5,0.,
                                  min(stop,time.monotonic()+5.),
                                  lambda sol:accept(sol,collect=False),stats)
                            stats["polishes"]+=1
                        except Paused:
                            break
                state["trials"].append({"plan":key,"deadline":due,"cost":time.monotonic()-t,
                                        "counts":dict(stats),"distinct":len(seen)})
                atomic_json(file,state)
                print(state["trials"][-1],flush=True)
                if time.monotonic()>=stop:
                    raise BudgetExpired
    except BudgetExpired:
        pass
    finally:
        state["wall_seconds"]=state.get("wall_seconds",0.)+time.monotonic()-start
        atomic_json(file,state)
    print("FINISH",state["best"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds",type=float,default=75.)
    p.add_argument("--trial-seconds",type=float,default=15.)
    p.add_argument("--starts",type=int,default=2)
    p.add_argument("--minimum-gain",type=float,default=.5,
                   help="Cap the requested gain per deadline probe in days")
    p.add_argument("--polish-limit",type=int,default=0)
    p.add_argument("--finish-slack",type=float,default=.12)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path)
    run(p.parse_args())
