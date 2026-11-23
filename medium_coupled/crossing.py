"""Profile prior terminal Lambert branches and reconstruct against their deadlines."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.run import ROOT, BASE
from medium_coupled.endgame import decode
from medium_overnight.validation import official_udp, verify, solution_payload
from medium_v4.driver import unpack
from medium_coupled.checkpoint import atomic_json
from pareto_easy import BudgetExpired


def run(args):
    start=time.monotonic()
    stop=start+args.seconds
    udp=official_udp(ROOT)
    seed=unpack(json.loads(args.seed_record.read_text()))
    source_score=verify(udp,seed)[0]
    state_file=args.directory/"checkpoint.json"
    state=json.loads(state_file.read_text()) if state_file.exists() else {
        "source":str(args.seed_record),"score":source_score,"record":None,
        "span":args.span,"windows":[],"fits":[],"repairs":[]}
    assert state["source"]==str(args.seed_record)
    assert state["span"]==args.span
    n=len(seed["dep"])
    edge=tuple(seed["path"][-2:])
    a,b=edge
    prefix={"path":seed["path"][:-1],"dep":seed["dep"][:-1],"tof":seed["tof"][:-1]}
    final_dep=seed["dep"][-1]
    count=Counter(state.get("counts",{}))
    def save():
        state["counts"]=dict(count)
        atomic_json(state_file,state)

    def accept(candidate, provenance):
        count["official_checks"]+=1
        try:
            fit=verify(udp,candidate)
        except ValueError:
            return False
        count["valid_schedules"]+=1
        if fit[0]<state["score"]:
            fit=verify(official_udp(ROOT),candidate)
            payload=solution_payload(candidate,fit,udp=udp,provenance=provenance)
            record=args.directory/"records"/f"{fit[0]:.12f}_{uuid4().hex[:8]}.json"
            atomic_json(record,{"solution":candidate,**payload})
            state.update(score=fit[0],record=str(record.resolve()))
            save()
            print("RECORD",fit,record,flush=True)
        return True

    try:
        for i in range(len(state["fits"]),args.windows):
            due=(source_score-.001) if i==0 else final_dep-(i-1)*.30+.10
            floor=max(0.,due-args.span)
            minima=[]
            for j,cap in enumerate((udp.dv_threshold,udp.dv_threshold_exception)):
                rows,cost=fit_transfer_window(udp,a,b,floor,due,cap,
                                              stop,random.Random(4321+2*i+j))
                minima.append(cost)
                for leg in rows:
                    if leg[2]+leg[3]<source_score-.001:
                        state["windows"].append([i,*leg])
            state["fits"].append({"release":floor,"due":due,"sampled_minima":minima})
            save()
        # Later departure is easier to join. Refresh all 179-leg prefix
        # states independently for each distinct terminal branch.
        branches=sorted(state["windows"],key=lambda r:-r[3])
        tried={tuple(r["key"]) for r in state["repairs"]}
        for i,row in enumerate(branches):
            _,a,b,d,f,_=row
            for width,step in ((3,.02),(6,.03125),(9,.025),(12,.025)):
                key=(i,width,step)
                if key in tried:
                    continue
                if time.monotonic()>=stop:
                    raise BudgetExpired
                begin=time.monotonic()
                stats=Counter()
                # Reserve this terminal event's actual class during prefix DP.
                spend=int(udp.compute_transfer(a,b,d,f)>udp.dv_threshold)
                def join(candidate):
                    if candidate["path"][-1]!=a:
                        return
                    full={"path":candidate["path"]+[b],"dep":candidate["dep"]+[d],
                          "tof":candidate["tof"]+[f]}
                    stats["joined"]+=1
                    accept(full,{"terminal_window":[d,f],"repair":key})
                try:
                    decode(udp,prefix,n-1-width,d,step,.007,min(stop,begin+args.trial_seconds),
                           stats,join, reserve=spend, endpoint=a)
                except BudgetExpired:
                    stats["time_cap"]=1
                    if time.monotonic()>=stop:
                        raise
                count.update(stats)
                state["repairs"].append({"key":key,"seconds":time.monotonic()-begin,
                                           "stats":dict(stats)})
                save()
                print("REPAIR",key,d,dict(stats),flush=True)
    except BudgetExpired:
        pass
    finally:
        state["wall_seconds"]=state.get("wall_seconds",0.)+time.monotonic()-start
        save()
    print("FINISH",state["score"],"windows",len(state["windows"]),dict(count),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds",type=float,default=120.)
    p.add_argument("--trial-seconds",type=float,default=12.)
    p.add_argument("--windows",type=int,default=18)
    p.add_argument("--span",type=float,default=2.1)
    p.add_argument("--seed-record",type=Path,default=BASE)
    p.add_argument("--directory",type=Path,required=True)
    run(p.parse_args())
