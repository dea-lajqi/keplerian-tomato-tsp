"""Move the terminal tomato into an earlier orbital passage and rejoin the suffix."""
import argparse
from collections import Counter
from itertools import pairwise
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import geometry
from medium_coupled.run import ROOT
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
    score=verify(udp,seed)[0]
    p,d,f=(seed[key] for key in ("path","dep","tof"))
    moved=p[-args.tail_size:]
    n=len(d)
    last=n-len(moved)
    spend=[int(udp.compute_transfer(a,b,t,q)>udp.dv_threshold)
           for (a,b),t,q in zip(pairwise(p),d,f)]
    (normal,allowed),floor=geometry(udp,4.9048695e12)
    plans=[]
    for width,step,phase in ((1,.03125,.007),(3,.0625,.013),(5,.0625,.034),
                             (2,.02,.004),(4,.03125,.012),(7,.0625,.045)):
        for left in range(1,last-width+1):
            right=left+width
            remaining=p[left+1:right]
            members=[p[left],*remaining,*moved,p[right]]
            reserve=sum(spend[right:last])+args.reserve
            available=udp.n_dv_exception-sum(spend[:left])-reserve
            # Every inserted vertex requires both an incoming and outgoing arc.
            possible=[]
            for target in moved:
                ins=[a for a in members[:-1] if a!=target and allowed[a,target]]
                outs=[b for b in members[1:] if b!=target and allowed[target,b]]
                pairs=[(a,b) for a in ins for b in outs if a!=b and
                       int(not normal[a,target])+int(not normal[target,b])<=available]
                if not pairs:
                    break
                possible.append(min(floor[a,target]+floor[target,b] for a,b in pairs))
            else:
                plans.append((width,sum(possible),left,right,step,phase,reserve))
    plans.sort()
    file=args.directory/"checkpoint.json"
    state=json.loads(file.read_text()) if file.exists() else {
        "baseline":score,"best":score,"record":None,"trials":[],
        "geometry_screened_windows":len(plans),"valid_route_signatures":[],
        "moved":moved,"seed":str(args.seed_record.resolve()),"reserve":args.reserve}
    if state.get("moved",moved)!=moved or state.get("seed",str(args.seed_record.resolve()))!=str(args.seed_record.resolve()):
        raise ValueError("Checkpoint belongs to a different seed or terminal block")
    valid_routes=set(state.get("valid_route_signatures",[]))
    done={tuple(r["key"]) for r in state["trials"]}
    try:
        for _,_,left,right,step,phase,reserve in plans:
            key=(left,right,step,phase)
            if key in done:
                continue
            if time.monotonic()>=stop:
                break
            stats=Counter()
            t=time.monotonic()
            prefix={"path":p[:right+1],"dep":d[:right],"tof":f[:right]}
            targets=[*p[left+1:right],*moved,p[right]]
            def submit(partial):
                candidate={"path":partial["path"]+p[right+1:-len(moved)],
                           "dep":partial["dep"]+d[right:last],
                           "tof":partial["tof"]+f[right:last]}
                stats["official_checks"]+=1
                try:
                    fitness=verify(udp,candidate)
                except ValueError:
                    return
                stats["valid_schedules"]+=1
                order=bytes(candidate["path"])
                valid_routes.add(hashlib.sha256(order).hexdigest())
                state["valid_route_signatures"]=sorted(valid_routes)
                if fitness[0]<state["best"]:
                    newfit=verify(official_udp(ROOT),candidate)
                    payload=solution_payload(candidate,newfit,udp=udp,
                                             provenance={"inserted":moved,"window":key})
                    record=args.directory/"records"/f"{newfit[0]:.12f}_{uuid4().hex[:8]}.json"
                    atomic_json(record,{"solution":candidate,**payload})
                    state.update(best=newfit[0],record=str(record.resolve()))
                    atomic_json(file,state)
                    print("RECORD",newfit,record,flush=True)
            try:
                decode(udp,prefix,left,d[right],step,phase,min(stop,t+args.trial_seconds),
                       stats,submit,reserve=reserve,endpoint=p[right],targets_override=targets)
            except BudgetExpired:
                stats["time_cap"]=1
                if time.monotonic()>=stop:
                    break
            state["trials"].append({"key":key,"seconds":time.monotonic()-t,"stats":dict(stats)})
            if len(state["trials"])%8==0 or stats["official_checks"]:
                atomic_json(file,state)
    finally:
        state["wall_seconds"]=state.get("wall_seconds",0.)+time.monotonic()-start
        atomic_json(file,state)
    print("FINISH",state["best"],len(state["trials"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds",type=float,default=120)
    parser.add_argument("--trial-seconds",type=float,default=6)
    parser.add_argument("--seed-record",type=Path,required=True)
    parser.add_argument("--tail-size",type=int,choices=(1,2,3),default=1)
    parser.add_argument("--reserve",type=int,choices=(0,1),default=0)
    parser.add_argument("--directory",type=Path,required=True)
    run(parser.parse_args())
