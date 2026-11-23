"""Remove a late tomato, rebuild the shorter finish, then repair an earlier window."""
import argparse
from collections import Counter
import json
from itertools import pairwise
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.endgame import decode
from medium_coupled.run import ROOT
from medium_overnight.validation import official_udp, solution_payload, verify
from medium_v4.driver import unpack
from medium_coupled.checkpoint import atomic_json
from global_event_easy import geometry
from pareto_easy import BudgetExpired


def run(args):
    start=time.monotonic()
    stop=start+args.seconds
    udp=official_udp(ROOT)
    original=unpack(json.loads(args.seed_record.read_text()))
    baseline=verify(udp,original)[0]
    file=args.directory/"checkpoint.json"
    state=json.loads(file.read_text()) if file.exists() else {
        "baseline":baseline,"best":baseline,"record":None,"trials":[],"route_signatures":[]}
    order=original["path"]
    (normal,allowed),floors=geometry(udp,4.9048695e12)
    count=Counter(state.get("counts",{}))
    done={tuple(x["plan"]) for x in state["trials"]}
    route_signatures=set(state["route_signatures"])

    def checkpoint():
        state["counts"]=dict(count)
        state["route_signatures"]=sorted(route_signatures)
        atomic_json(file,state)

    def restore(short,cut,moved,stats,deadline,plan):
        path,dep,tof=(short[k] for k in ("path","dep","tof"))
        e=[int(udp.compute_transfer(a,b,d,f)>udp.dv_threshold)
           for (a,b),d,f in zip(pairwise(path),dep,tof)]
        sites=[]
        for width in (1,2,3):
            for left in range(max(1,cut-32),cut-width):
                right=left+width
                room=dep[right]-dep[left-1]-tof[left-1]
                budget=udp.n_dv_exception-sum(e[:left])-sum(e[right:])
                candidates=[floors[a,moved]+floors[moved,b]
                            for a in path[left:right] for b in path[left+1:right+1]
                            if a!=b and allowed[a,moved] and allowed[moved,b]
                            and int(not normal[a,moved])+int(not normal[moved,b])<=budget]
                if candidates and room>(width+1)*udp.min_tof:
                    sites.append((width,min(candidates)/(room+.05),left,right))
        for _,_,left,right in sorted(sites)[:10]:
            stats["earlier_windows"]+=1
            def validate(part):
                candidate={"path":part["path"]+path[right+1:],
                           "dep":part["dep"]+dep[right:],"tof":part["tof"]+tof[right:]}
                stats["official_checks"]+=1
                try:
                    fit=verify(udp,candidate)
                except ValueError:
                    return
                stats["valid_schedules"]+=1
                route_signatures.add(bytes(candidate["path"]).hex())
                if fit[0]<state["best"]:
                    independent=verify(official_udp(ROOT),candidate)
                    payload=solution_payload(candidate,independent,udp=udp,
                                provenance={"plan":plan,"insert":[left,right],"moved":moved})
                    target=args.directory/"records"/f"{fit[0]:.12f}_{uuid4().hex[:8]}.json"
                    atomic_json(target,{"solution":candidate,**payload})
                    state.update(best=fit[0],record=str(target.resolve()))
                    checkpoint()
                    print("RECORD",fit,target,flush=True)
            decode(udp,short,left,dep[right],.025,.006,deadline,stats,validate,
                   reserve=sum(e[right:]),endpoint=path[right],
                   targets_override=[*path[left+1:right],moved,path[right]])

    try:
        for j in range(len(order)-2,len(order)-10,-1):
            for extra,step in ((0,.025),(2,.03125)):
                plan=(j,extra,step)
                if plan in done:
                    continue
                if time.monotonic()>=stop:
                    raise BudgetExpired
                cut=j-1-extra
                stats=Counter()
                now=time.monotonic()
                local_stop=min(stop,now+args.trial_seconds)
                seen=set()

                def join(partial):
                    if len(seen)>=5:
                        return
                    signature=tuple(partial["path"])
                    if signature not in seen:
                        seen.add(signature)
                        stats["shortened_suffixes"]+=1
                        restore(partial,cut,order[j],stats,local_stop,plan)
                try:
                    decode(udp,original,cut,state["best"]-.005,step,.004,
                           local_stop,stats,join,targets_override=order[cut+1:j]+order[j+1:])
                except BudgetExpired:
                    stats["time_cap"]=1
                count.update(stats)
                state["trials"].append({"plan":plan,"seconds":time.monotonic()-now,"stats":dict(stats)})
                checkpoint()
    except BudgetExpired:
        pass
    finally:
        state["wall_seconds"]=state.get("wall_seconds",0.)+time.monotonic()-start
        checkpoint()
    print("FINISH",state["best"],dict(count),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds",type=float,default=110)
    p.add_argument("--trial-seconds",type=float,default=8)
    p.add_argument("--seed-record",required=True,type=Path)
    p.add_argument("--directory",required=True,type=Path)
    run(p.parse_args())
