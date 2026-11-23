"""Paired diagnostics for Small timing bounds, cache dominance and phase stopping."""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import focus_easy
from continuous_windows_easy import FILES, SEED
from elite_easy import anchor_bank, scan_archive
from global_dp_easy import Grid, evaluator
from kttsp_esa_solver import validate
from pareto_easy import arrival, BudgetExpired
from small_v2 import atomic_json


class Results:
    def __init__(self,udp):
        self.udp=udp
        self.checks=0
        self.feasible=0
        self.best=None
        self.orders=set()
        self.seen=set()

    def add(self,path,legs):
        key=tuple(tuple(l[:4]) for l in legs)
        if key in self.seen:
            return
        self.seen.add(key)
        self.checks+=1
        result=validate(self.udp,path,legs)
        if result and result["valid"]:
            self.feasible+=1
            self.orders.add(tuple(path))
            if self.best is None or result["fitness"][0]<self.best["validation"]["fitness"][0]:
                self.best={"path":path,"transfers":legs,"validation":result}

    def report(self):
        check=(validate(evaluator().udp,self.best["path"],self.best["transfers"])
               if self.best else None)
        return {"checks":self.checks,"feasible":self.feasible,"orders":len(self.orders),
                "independent":check,"best_schedule":self.best}


def corpus(udp):
    seed=json.loads(SEED.read_text())
    campaigns=FILES/"small_v10"
    records={tuple(seed["path"]):seed}
    events=[]
    for directory in ("learned_replay","matched_control","event_feedback","two_directions"):
        files=sorted((campaigns/directory).glob("*checkpoint.json"),
                     key=lambda p:p.stat().st_mtime_ns)
        if not files:
            continue
        for f in files:
            d=json.loads(f.read_text())
            events.extend(tuple(l) for l in d.get("learned_events",()))
        d=json.loads(files[-1].read_text())
        for row in d["pool"]:
            p=tuple(row["path"])
            value=arrival(row["transfers"][-1])
            if p not in records or value<arrival(records[p]["transfers"][-1]):
                records[p]=row
    candidates=sorted(records.values(),key=lambda r:arrival(r["transfers"][-1]))
    # Include a small set of close orders and a substantially different start.
    reference=tuple(seed["path"])
    short=[seed]
    short.extend(r for r in candidates if tuple(r["path"])!=reference and
                 r["path"][0]==reference[0])
    short=short[:5]
    short.extend([r for r in candidates if r["path"][0]!=reference[0]][:4])
    events.extend(tuple(l) for r in candidates for l in r["transfers"])
    return seed,short,events


def bounds(module,seed):
    """Run the same implementation changing only the coordinate box."""
    source=inspect.getsource(focus_easy.refine_joint)
    margin="start_margin = np.tile([.55, .22], n)"
    if source.count(margin)!=1:
        raise RuntimeError("Unexpected incumbent refiner source")
    out={}
    for label,dep,tof in (("default",.55,.22),("wide",3.,2.)):
        namespace=dict(vars(focus_easy))
        text=source.replace(margin,f"start_margin = np.tile([{dep!r}, {tof!r}], n)")
        text=text.replace("    accept(fit.x)","    accept(fit.x)\n    return fit")
        exec(compile(text,"<local bounds diagnostic>","exec"),namespace)
        refiner=namespace["refine_joint"]
        start=time.monotonic()
        results=Results(module.udp)
        # The input is already valid; keep it as the incumbent fallback
        # even when the solver returns no qualifying iterate.
        results.add(seed["path"],seed["transfers"])
        fits=[]
        for k in (0,1,2):
            try:
                variations=tuple(map(tuple,seed["transfers"]))
                fit=refiner(module.udp,variations,start+42.,
                        lambda l:results.add(seed["path"],l),
                        0.,random.Random(3601+k),bool(k))
                fits.append({"status":int(fit.status),"iterations":int(fit.nit),
                             "message":str(fit.message),"raw_objective":float(fit.fun)})
            except BudgetExpired:
                fits.append({"budget_hit":True})
                break
        compute=time.monotonic()-start
        out[label]={"compute_seconds":compute,"fits":fits,**results.report()}
    return out


def learned_cache(events,udp,pareto):
    buckets=defaultdict(list)
    for l in events:
        a,b,d,f,dv=l
        group=(a,b,round(d*4),int(dv>udp.dv_threshold))
        saved=buckets[group]
        if not pareto:
            if not saved or arrival(l)<arrival(saved[0]):
                buckets[group]=[l]
        elif not any(other[2]>=d and arrival(other)<=arrival(l) for other in saved):
            buckets[group]=[other for other in saved if not (
                d>=other[2] and arrival(l)<=arrival(other))]+[l]
    return buckets


def cache_test(module,seed,orders,events):
    archive,_=scan_archive(module.udp)
    shared=anchor_bank(archive)
    # Deduplicate and recompute arc feasibility from the original transfer routine.
    verified=[]
    keys=set()
    begin=time.monotonic()
    for l in events:
        key=tuple(l[:4])
        if key in keys:
            continue
        keys.add(key)
        a,b,d,f,_=l
        dv=float(module.udp.compute_transfer(a,b,d,f))
        if dv<=module.udp.dv_threshold_exception+1e-6:
            verified.append((a,b,d,f,dv))
    out={"verified_transfer_inputs":len(verified),"shared_remeasurement_seconds":time.monotonic()-begin}
    arms={}
    for label,fix in (("old",False),("pareto",True)):
        start=time.monotonic()
        buckets=learned_cache(verified,module.udp,fix)
        by_pair=defaultdict(set,{pair:set(legs) for pair,legs in shared.items()})
        for group,legs in buckets.items():
            by_pair[group[:2]].update(legs)
        anchors={pair:tuple(sorted(rows)) for pair,rows in by_pair.items()}
        cache_seconds=time.monotonic()-start
        results=Results(module.udp)
        results.add(seed["path"],seed["transfers"])
        grid=Grid(module,.0625,.013,start+65.,directional=True)
        grid.horizon=132.
        complete=[]
        try:
            for r in orders:
                path=tuple(r["path"])
                schedules,depth=focus_easy.retime(grid,path,anchors,wait=5.25,flight=10.)
                for schedule in schedules:
                    results.add(path,schedule)
                complete.append({"path":path,"depth":depth,
                                 "finish":min((arrival(s[-1]) for s in schedules),default=None)})
        except BudgetExpired:
            pass
        arms[label]=buckets
        compute=time.monotonic()-start
        out[label]={"cache_seconds":cache_seconds,"compute_seconds":compute,
                    "cached_opportunities":sum(map(len,buckets.values())),
                    "per_order":complete,**results.report()}
    lost=[]
    for key,rows in arms["pareto"].items():
        old=arms["old"][key][0]
        for l in rows:
            if l[2]>old[2] and arrival(l)>arrival(old):
                lost.append({"cached_earlier_arrival":old,"retained_later_departure":l,
                             "usable_at_release":(l[2]+old[2])/2})
    out["release_availability_witnesses"]=len(lost)
    out["witnesses"]=sorted(lost,key=lambda x:-(
        x["retained_later_departure"][2]-x["cached_earlier_arrival"][2]))[:5]
    return out


def phases(module,seed,orders):
    archive,_=scan_archive(module.udp)
    anchors=anchor_bank(archive)
    results={k:Results(module.udp) for k in ("stop_at_first_complete","always_try_phase_two")}
    for r in results.values():
        r.add(seed["path"],seed["transfers"])
    start=time.monotonic()
    grids=[Grid(module,.0625,.011,start+120.,directional=True),
           Grid(module,.046875,.022,start+120.,directional=True)]
    for grid in grids:
        grid.horizon=132.
    by_order=[]
    totals=defaultdict(float)
    for row in orders:
        path=tuple(row["path"])
        trial={"path":path}
        for index,grid in enumerate(grids):
            t=time.monotonic()
            try:
                options,depth=focus_easy.retime(grid,path,anchors,wait=5.25,flight=10.)
            except BudgetExpired:
                trial["budget_hit"]=True
                by_order.append(trial)
                options=[]
                break
            for solution in options:
                results["always_try_phase_two"].add(path,solution)
                if index==0 or trial["phase_1"]["best"] is None:
                    results["stop_at_first_complete"].add(path,solution)
            delta=time.monotonic()-t
            totals["always_try_phase_two"]+=delta
            if index==0 or trial["phase_1"]["best"] is None:
                totals["stop_at_first_complete"]+=delta
            trial[f"phase_{index+1}"]={"seconds":delta,"depth":depth,
                         "best":min((arrival(l[-1]) for l in options),default=None)}
        if trial.get("budget_hit"):
            break
        by_order.append(trial)
    return {"per_order":by_order,**{k:{"compute_seconds":totals[k],**v.report()}
                                    for k,v in results.items()}}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("bounds","cache","phases"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    mod=evaluator()
    seed,orders,events=corpus(mod.udp)
    result={"bounds":lambda:bounds(mod,seed),"cache":lambda:cache_test(mod,seed,orders,events),
            "phases":lambda:phases(mod,seed,orders)}[args.mode]()
    atomic_json(args.output,result)
    for k,v in result.items():
        if isinstance(v,dict) and "independent" in v:
            print(k,v["compute_seconds"],v["independent"],v["feasible"],flush=True)
