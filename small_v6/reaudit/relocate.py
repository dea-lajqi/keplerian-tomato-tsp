"""Test whether a late exception can replace an upstream exception in Small."""
import argparse
import json
from pathlib import Path
import random
import signal
import sys
import time
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from continuous_windows_easy import FILES, SEED
from elite_easy import anchor_bank, scan_archive
from focus_easy import retime
from global_dp_easy import Grid, evaluator
from global_event_easy import Topology, geometry
from kttsp_esa_solver import chromosome, validate
from pareto_easy import BudgetExpired, arrival, write_json
from small_v2 import atomic_json


def run(args):
    started=time.monotonic()
    deadline=started+args.seconds
    module=evaluator()
    udp=module.udp
    allowed,floors=geometry(udp,module.MU_MOON)
    archive,_=scan_archive(udp)
    anchors=anchor_bank(archive)
    raw=json.loads(SEED.read_text())
    baseline=validate(udp,raw["path"],raw["transfers"])
    assert baseline and baseline["valid"]
    ref=set(zip(raw["path"],raw["path"][1:]))
    best=baseline["fitness"][0]
    extra=(24,6)
    topology=Topology(udp,allowed,floors,archive)
    rng=random.Random(args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    run_id=uuid4().hex[:8]
    hist=[]
    validated={}
    attempts=0
    def checkpoint():
        atomic_json(args.output/f"relocation_{run_id}_checkpoint.json",
            {"baseline":baseline,"best_days":best,"attempts":attempts,"elapsed":time.monotonic()-started,
             "history":hist,"valid_candidates":list(validated.values()),"seed":args.seed})
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        # Reuse archived orders that already contain the desired transfer.
        known=[p for p in sorted(archive,key=lambda p:min(row[0] for row in archive[p]))
               if extra in set(zip(p,p[1:]))]
        for i in range(55):
            if time.monotonic()>=deadline:
                break
            attempts+=1
            if i<min(6,len(known)):
                route=known[i]
            else:
                end=raw["path"][-1] if i%2 else 32
                n=udp.n_tomatoes
                fixed=((n,raw["path"][0]),(end,n))
                ans=topology.propose(rng,min(deadline,time.monotonic()+4.),ref,
                    4,1.8,forced=fixed,forced_exception=extra)
                if not ans:
                    continue
                route=ans[0]
            edges=tuple(zip(route,route[1:]))
            needed=sum(not allowed[0][a,b] or (a,b)==extra for a,b in edges)
            if needed>udp.n_dv_exception:
                continue
            trial={"path":route,"screened_exceptions":needed,"depth":0}
            hist.append(trial)
            for step,phase in ((.0625,.021),(.046875,.012)):
                grid=Grid(module,step,phase,deadline,directional=True)
                grid.horizon=132.
                schedules,depth=retime(grid,route,anchors,extra_edge=extra,require_extra=extra,
                                      wait=5.,flight=9.5)
                trial["depth"]=max(depth,trial["depth"])
                for legs in schedules:
                    check=validate(udp,route,legs)
                    if not check or not check["valid"]:
                        continue
                    score=check["fitness"][0]
                    record={"path":route,"transfers":legs,"validation":check,
                            "chromosome":chromosome(route,legs),"seed":args.seed,"late_exception":extra}
                    if route not in validated or score<validated[route]["validation"]["fitness"][0]:
                        validated[route]=record
                    if score<best:
                        independent=validate(evaluator().udp,route,legs)
                        if independent and independent["valid"] and independent["fitness"][0]<best:
                            best=independent["fitness"][0]
                            write_json(args.output/f"easy_relocated_{run_id}_{len(hist):03}.json",record)
                if schedules:
                    break
            if i%6==0:
                checkpoint()
    except (BudgetExpired,KeyboardInterrupt):
        pass
    finally:
        checkpoint()
        print("attempts",attempts,"timed orders",len(hist),"validated orders",len(validated),
              "best",best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=65.)
    p.add_argument("--seed",type=int,default=86219)
    run(p.parse_args())
